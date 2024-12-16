from datetime import datetime
from arckit.data import Task
from arc import settings
from transformers import TrainingArguments

from trl import SFTTrainer
from unsloth import FastLanguageModel
from arc import utils
from arc.datasets.barc_modified_programs import get_finetune_dataset
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

# global set up
# dataset = get_raw_dataset()["train"]
# model, tokenizer = FastLanguageModel.from_pretrained(
#     "barc0/Llama-3.1-ARC-Potpourri-Induction-8B",
#     dtype=torch.bfloat16,
# )
# model = FastLanguageModel.for_inference(model)


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


# Some data  validation
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
    outputs = model.generate(**inputs, temperature=0.8, num_return_sequences=1)
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
    batch_output = model.generate(**batch_input)
    input_length = batch_input["input_ids"].size(1)
    decoded_batch_output = [
        tokenizer.decode(o[input_length:], skip_special_tokens=True)
        for o in batch_output
    ]
    return [Program.from_source(parse_code(code)) for code in decoded_batch_output]


########
def get_baseline_program_improvement():
    SAMPLE_COUNT = 100
    sampled_dataset = dataset.shuffle().select(range(SAMPLE_COUNT))
    tasks = [Task(**r["task"]) for r in sampled_dataset]
    # we strip the comments to remove hints
    initial_programs = [
        Program.from_source(remove_comments(r["modified_program_source"]))
        for r in sampled_dataset
    ]
    executions = [
        ProgramExecution(program, task)
        for task, program in zip(tasks, initial_programs)
    ]
    improved_programs = []
    for i, batch_execution in tqdm(
        enumerate(utils.batch(executions, INFERENCE_BATCH_SIZE)),
        total=SAMPLE_COUNT // INFERENCE_BATCH_SIZE,
    ):
        try:
            improved_programs.extend(
                batch_generate_improved_program(model, tokenizer, batch_execution)
            )
        except Exception:
            logger.exception(f"failed batch {i}")

    for i, (task, program) in enumerate(zip(tasks, improved_programs)):
        print(evaluate_program(program, task, i))


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
            num_train_epochs=3,  # Set this for 1 full training run.
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
