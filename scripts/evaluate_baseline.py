import arckit

import torch
from arc.tasks import prompts
import re
from rich.console import Console
from rich.table import Table
from typing import Callable, Optional
import os
import numpy as np
from loguru import logger
from unsloth import FastLanguageModel
from arc.types import Program


#  loading in model
CHECKPOINT_DIR = (
    "./tmp/runs/llama_3_1_8b_lora_barc_finetune_20241114_193633/checkpoint-12123/"
)
SOLUTION_DIR = (
    "data/runs/llama_3_1_8b_lora_barc_finetune_20241114_193633/code_generation"
)

# model, tokenizer = FastLanguageModel.from_pretrained(
#     CHECKPOINT_DIR,
#     dtype=torch.bfloat16,
# )
# model, tokenizer = FastLanguageModel.from_pretrained(
#     "barc0/Llama-3.1-ARC-Heavy-Induction-8B",
#     dtype=torch.bfloat16,
# )
model, tokenizer = FastLanguageModel.from_pretrained(
    "barc0/Llama-3.1-ARC-Potpourri-Induction-8B",
    dtype=torch.bfloat16,
)


model = FastLanguageModel.for_inference(model)


os.makedirs(SOLUTION_DIR, exist_ok=True)


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


def batch_generate_code(model, tokenizer, task, num_samples: int) -> list[str]:
    messages = [
        dict(role="system", content=prompts.programmer_role_prompt),
        dict(role="user", content=prompts.create_solve_task_prompt(task)),
    ]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_text, return_tensors="pt")
    outputs = model.generate(
        **inputs, temperature=0.8, num_return_sequences=num_samples
    )
    input_length = inputs["input_ids"].size(1)
    decoded_responses = [
        tokenizer.decode(o[input_length:], skip_special_tokens=True) for o in outputs
    ]
    codes = [parse_code(x) for x in decoded_responses]
    return [x for x in codes if x is not None]


def generate_improve_code(model, tokenizer, task, programs: list[Program]) -> str:
    messages = [
        dict(role="system", content=prompts.programmer_role_prompt),
        dict(
            role="user",
            content=prompts.create_improve_solve_task_prompt(task, programs),
        ),
    ]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_text, return_tensors="pt")
    outputs = model.generate(**inputs, temperature=0.8, num_return_sequences=1)
    input_length = inputs["input_ids"].size(1)
    decoded_responses = [
        tokenizer.decode(o[input_length:], skip_special_tokens=True) for o in outputs
    ]
    codes = [parse_code(x) for x in decoded_responses]
    return codes[0]


def interpret_code(source_code: str) -> Callable:
    local = dict()
    exec(source_code, local)
    assert "transform" in local
    return local["transform"]


def log_output(expected, actual):
    console = Console()
    table = Table()
    table.add_row(*[arckit.fmt_grid(x) for x in actual])
    table.add_row()
    table.add_row(*[arckit.fmt_grid(x) for x in expected])
    console.print(table)


def verify_code(task: arckit.Task, source_code: str) -> Callable | None:
    fn = interpret_code(source_code)
    if fn is not None:
        train_inputs, train_outputs = zip(*task.train)
        try:
            actual_train_outputs = [fn(i) for i in train_inputs]
            log_output(train_outputs, actual_train_outputs)
            if not any(
                [
                    actual.shape == expected.shape and np.all(actual == expected)
                    for actual, expected in zip(actual_train_outputs, train_outputs)
                ]
            ):
                fn = None
        except Exception as e:
            logger.exception(f"failed to invoke fn for at least 1 input {e}")
            fn = None
    return fn


def solve_task(
    model,
    tokenizer,
    task: arckit.Task,
    max_attempts: int = 1_000,
    save_fname: Optional[str] = None,
):
    batch_size = 2
    for i in range(0, max_attempts, batch_size):
        logger.debug(f"on attempt {i}")
        source_codes = batch_generate_code(model, tokenizer, task, batch_size)
        for source_code in source_codes:
            fn = verify_code(task, source_code)
            if fn is not None:
                if save_fname:
                    with open(save_fname, "w") as f:
                        f.write(source_code)
                return fn, source_code


def generate_program(model, tokenizer, task: arckit.Task) -> Program:
    sources = batch_generate_code(model, tokenizer, task, 1)
    assert len(sources) == 1
    source = sources[0]
    fn = interpret_code(source)
    return Program(task, source, fn)


def improve_program(
    model, tokenizer, task: arckit.Task, programs: list[Program]
) -> list[Program]:
    source = generate_improve_code(model, tokenizer, task, programs)
    fn = interpret_code(source)
    return programs + [Program(task, source, fn)]


train_set, eval_set = arckit.load_data()

task_id = "f3cdc58f"
task = eval_set[task_id]
# result = solve_task(model, task)
# if result is not None:
#     fn, source_code = result
