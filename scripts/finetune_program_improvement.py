from datetime import datetime
import re
from arckit.data import Task
from datasets import Dataset
from arc import settings
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM

from trl import SFTTrainer
from unsloth import FastLanguageModel
from arc import utils
from arc.datasets.barc_modified_programs import get_finetune_dataset, get_raw_dataset
from arc.tasks import prompts
from arc.program import Program, ProgramExecution, remove_comments
from dataclasses import dataclass
import numpy as np
from concurrent.futures import Future, ProcessPoolExecutor
import multiprocessing
from typing import Tuple, Dict, Optional
from loguru import logger
import torch
from tqdm import tqdm

INFERENCE_BATCH_SIZE = 4


@dataclass
class EvaluationSummary:
    train_correct: int
    train_total: int


def evaluate_program(program: Program, task: Task, i: int | None) -> EvaluationSummary:
    train_correct = 0
    for input_, output in task.train:
        program_output = program.call(input_)
        if not isinstance(program_output, Exception):
            assert program_output is not None, i

            if program_output.shape == output.shape and np.all(
                program_output == output
            ):
                train_correct += 1
    return EvaluationSummary(train_correct=train_correct, train_total=len(task.train))


# Some data validation
def parse_code(response: str) -> Optional[str]:
    # Extract code between Python code blocks
    code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)

    if code_match is None:
        return None
    else:
        return (
            code_match.group(1)
            .strip()
            .replace("from common import", "from arc.dsl.common import")
        )


def generate_improved_program(
    model, tokenizer, task, executions: list[ProgramExecution]
) -> Program:
    messages = [
        dict(role="system", content=prompts.programmer_role_prompt),
        dict(
            role="user",
            content=prompts.create_improve_solve_task_prompt(task, executions),
        ),
    ]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to(device="cuda")
    outputs = model.generate(**inputs, num_return_sequences=1)
    input_length = inputs["input_ids"].size(1)
    decoded_responses = [
        tokenizer.decode(o[input_length:], skip_special_tokens=True) for o in outputs
    ]
    return Program.from_source(parse_code(decoded_responses[0]))


def batch_generate_improved_program(
    model,
    tokenizer,
    batch_execution: list[ProgramExecution],
):
    batch_text = []
    for execution in batch_execution:
        messages = [
            dict(role="system", content=prompts.programmer_role_prompt),
            dict(
                role="user",
                content=prompts.create_improve_solve_task_prompt(
                    execution.task, [execution]
                ),
            ),
        ]
        batch_text.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    batch_input = tokenizer(batch_text, return_tensors="pt", padding=True).to(
        device="cuda"
    )
    # TODO the max_new_tokens hardcode is bad
    batch_output = model.generate(**batch_input, max_new_tokens=2_000)
    input_length = batch_input["input_ids"].size(1)
    decoded_batch_output = [
        tokenizer.decode(o[input_length:], skip_special_tokens=True)
        for o in batch_output
    ]
    logger.debug(decoded_batch_output)
    return [Program.from_source(parse_code(code)) for code in decoded_batch_output]


def evaluate_program_improvement(
    model,
    tokenizer,
    dataset,
    batch_size: int,
):
    tasks = [Task(**r["task"]) for r in dataset]
    # we strip the comments to remove hints
    initial_programs = [
        Program.from_source(remove_comments(r["modified_program_source"]))
        for r in dataset
    ]
    executions = [
        ProgramExecution(program, task)
        for task, program in zip(tasks, initial_programs)
    ]
    improved_programs = []
    for i, batch_execution in tqdm(
        enumerate(utils.batch(executions, batch_size)),
        total=len(dataset) // batch_size,
    ):
        try:
            improved_programs.extend(
                batch_generate_improved_program(model, tokenizer, batch_execution)
            )
        except Exception:
            logger.exception(f"failed batch {i}")
    results = []
    for i, (task, program) in enumerate(zip(tasks, improved_programs)):
        summary = evaluate_program(program, task, i)
        logger.info(f"{i}: {summary}")
        results.append((program, task, summary))
    return results


def get_baseline_program_improvement():
    model, tokenizer = FastLanguageModel.from_pretrained(
        "barc0/Llama-3.1-ARC-Potpourri-Induction-8B",
        dtype=torch.bfloat16,
    )
    model = FastLanguageModel.for_inference(model)
    dataset = get_raw_dataset()["train"]
    sampled_dataset = dataset.shuffle().select(range(100))
    return evaluate_program_improvement(
        model,
        tokenizer,
        sampled_dataset,
        batch_size=INFERENCE_BATCH_SIZE,
    )


#####


def validate_row(i, row: Tuple[int, Dict]) -> int:
    task = Task(**row["task"])
    original_program = Program.from_source(row["original_program_source"])
    modified_program = Program.from_source(row["modified_program_source"])

    # ensuring programs changed
    assert original_program.source != modified_program.source

    # ensuring original program is right
    original_evaluation = evaluate_program(original_program, task, i)
    assert (
        original_evaluation.train_total == original_evaluation.train_correct
    ), f"original evaluation failed {i}"

    # ensuring original and modified do not match 100
    matches = True
    for input_, output in task.train:
        original_program_output = original_program.call(input_)
        modified_program_output = modified_program.call(input_)
        if (
            isinstance(original_program_output, np.ndarray)
            and isinstance(modified_program_output, np.ndarray)
            and original_program_output.shape == modified_program_output.shape
            and np.all(original_program_output == modified_program_output)
        ):
            pass
        else:
            matches = False
            break
    assert not matches, f"original and modified program matches {i}"


def validate_dataset():
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures: list[Future] = [
            executor.submit(validate_row, i, r) for i, r in enumerate(dataset)
        ]
    for future in futures:
        try:
            future.result()
        except Exception as e:
            logger.error(e)


########


def finetune_model():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"barc_program_improvement_2k_finetune_{timestamp}"
    output_dir = f"tmp/runs/{run_name}"

    dataset = get_finetune_dataset()["train"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        "barc0/Llama-3.1-ARC-Potpourri-Induction-8B",
        # dtype=torch.bfloat16,
        # "unsloth/Meta-Llama-3.1-8B-Instruct",
        dtype=torch.bfloat16,
        token=settings.HF_API_TOKEN,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing=True,
        random_state=3407,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=TrainingArguments(
            run_name=run_name,
            output_dir=output_dir,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            warmup_steps=0,
            num_train_epochs=1,  # Set this for 1 full training run.
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_torch_fused",
            save_strategy="epoch",
            weight_decay=0.0,
            lr_scheduler_type="cosine",
            seed=3407,
            report_to="wandb",
        ),
    )

    trainer.train()


#####


def evaluate_ft_model():
    # model = AutoModelForCausalLM.from_pretrained(
    #     "barc0/Llama-3.1-ARC-Potpourri-Induction-8B",
    #     attn_implementation="flash_attention_2",
    #     torch_dtype=torch.bfloat16,
    #     device_map="cuda",
    # )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "barc0/Llama-3.1-ARC-Potpourri-Induction-8B",
    # )

    ckpt_path = (
        "tmp/runs/barc_program_improvement_2k_finetune_20241211_220113/checkpoint-375"
    )
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    model.eval()
    model = torch.compile(model)

    dataset = get_raw_dataset()["train"]
    ft_dataset = get_finetune_dataset()["train"]
    sampled_dataset = dataset.shuffle().select(range(100))
    sampled_dataset = Dataset.from_dict(dataset[:10])

    results = evaluate_program_improvement(
        model,
        tokenizer,
        sampled_dataset,
        batch_size=2,
    )

    msg = ""
    for i, (row, ft_row, (improved_program, task, _)) in enumerate(
        zip(dataset, ft_dataset, results)
    ):
        original_program = Program.from_source(row["original_program_source"])
        modified_program = Program.from_source(row["modified_program_source"])
        msg += f"# id: {i}\n"
        msg += f"# ORIGINAL\n{original_program.source}\n"
        msg += f"# MODIFIED\n{modified_program.source}\n"
        msg += f"# IMPROVED\n{improved_program.source}\n"
        msg += "\n\n"
