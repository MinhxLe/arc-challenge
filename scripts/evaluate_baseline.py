import arckit

from arc.tasks import prompts
import re
from rich.console import Console
from rich.table import Table
from typing import Callable, Optional
import os
import torch
import numpy as np
from loguru import logger

from unsloth import FastLanguageModel


#  loading in model
CHECKPOINT_DIR = (
    "./tmp/runs/llama_3_1_8b_lora_barc_finetune_20241114_193633/checkpoint-12123/"
)
SOLUTION_DIR = (
    "data/runs/llama_3_1_8b_lora_barc_finetune_20241114_193633/code_generation"
)

model, tokenizer = FastLanguageModel.from_pretrained(
    CHECKPOINT_DIR,
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


def batch_generate_code(task, num_samples: int) -> list[str]:
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


def interpret_code(source_code: str) -> Callable | None:
    try:
        local = dict()
        exec(source_code, local)
        assert "transform" in local
        return local["transform"]
    except Exception:
        logger.exception("failed to interpret code")
        return None


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
    task: arckit.Task,
    max_attempts: int = 1_000,
    save_fname: Optional[str] = None,
):
    batch_size = 4
    for i in range(0, max_attempts, batch_size):
        logger.debug(f"on attempt {i}")
        source_codes = batch_generate_code(task, batch_size)
        for source_code in source_codes:
            fn = verify_code(task, source_code)
            if fn is not None:
                if save_fname:
                    with open(save_fname, "w") as f:
                        f.write(source_code)
                return fn


train_set, eval_set = arckit.load_data()

solve_task(eval_set[0])

task = eval_set[0]
fn = solve_task(task)
#
# task_to_transform_fn = dict()
# # generating code
# for i, task in enumerate(eval_set):
#     print(f"starting task {task.id}")
#     code = generate_code(task)
#     if code is None:
#         print(f"failed to generate code for {task.id}")
#     else:
#         with open(f"{SOLUTION_DIR}/solution_{task.id}.py", "w") as f:
#             f.write(code)
#         try:
#             fn = interpret_code(code)
#             task_to_transform_fn[task.id] = fn
#         except Exception:
#             print(f"failed to interpret code for {task.id}")
#
# correct_tasks = []
# to_skip = [
#     "08573cc6",  # infinite loop?
# ]
# for i, task in enumerate(eval_set):
#     task_id = task.id
#     print(f"checking {task.id}")
#     if task_id in to_skip:
#         continue
#     if task_id in task_to_transform_fn:
#         input, expected_output = task.test[0]
#         try:
#             output = task_to_transform_fn[task.id](input)
#             if all(output == expected_output):
#                 correct_tasks.append(task_id)
#         except Exception:
#             print(f"failed to invoke transform for {task_id}")
#     else:
#         print(f"no fn for {task_id}")
#
#
# def print_evaluation(output, expected):
#     console = Console()
#     table = Table()
#     table.add_row(fmt_grid(output), fmt_grid(expected))
#     console.print(table)
#
#
# def print_grid(grid):
#     console = Console()
#     table = Table()
#     table.add_row(fmt_grid(grid))
#     console.print(table)


# def debug_task(task):
#             print_grid(output)
#     print(f"evaluating {task.id}")
#     task.show(answer=True)
#     if task.id in task_to_transform_fn:
#         input, expected_output = task.test[0]
#         try:
#             output = task_to_transform_fn[task.id](input)
#             print_grid(output)
#             print(
#                 f"got {np.sum(output == expected_output)}/{ np.prod(expected_output.shape)} correct"
#             )
#         except Exception as e:
#             print(f"failed to invoke transform {e}")
#         pass
#     else:
#         print("no fn found")
#     return None
