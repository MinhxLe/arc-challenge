import arckit

from arc.tasks import prompts
import re
from rich.console import Console
from typing import Callable, Optional
import os
from loguru import logger

from arc.external import openai
from arc.types import Program

SOLUTION_DIR = "data/openai/code_generation"
os.makedirs(SOLUTION_DIR, exist_ok=True)


def _modified_create_solve_task_prompt(task: arckit.Task) -> str:
    return prompts.create_solve_task_prompt(task) + prompts.addendum_for_nonfinetuned


def _modified_create_improve_solve_task_prompt(task, programs) -> str:
    return (
        prompts.create_improve_solve_task_prompt(task, programs)
        + f"Reminder: {prompts.addendum_for_nonfinetuned}"
    )


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


def generate_code(task) -> list[str]:
    codes = [
        parse_code(
            openai.complete(
                system_prompt=prompts.programmer_role_prompt,
                prompt=_modified_create_solve_task_prompt(task),
            )
        )
    ]
    return [x for x in codes if x is not None]


def generate_improve_code(task, programs: list[Program]) -> list[str]:
    codes = [
        parse_code(
            openai.complete(
                system_prompt=prompts.programmer_role_prompt,
                prompt=_modified_create_improve_solve_task_prompt(task, programs),
            )
        )
    ]
    return [x for x in codes if x is not None]


def interpret_code(source_code: str) -> Callable:
    local = dict()
    exec(source_code, local)
    assert "transform" in local
    return local["transform"]


def solve_task(
    task: arckit.Task,
    max_attempts: int = 1_000,
    save_fname: Optional[str] = None,
) -> list[Program]:
    programs = []
    console = Console(record=True)
    if save_fname:
        file_path = os.path.join(SOLUTION_DIR, save_fname + ".html")
        if os.path.exists(file_path):
            os.unlink(file_path)
        console = Console(record=True)
    else:
        file_path = None
        console = Console()

    for i in range(0, max_attempts):
        logger.debug(f"on attempt {i}")
        if len(programs) == 0:
            source_codes = generate_code(task)
        else:
            source_codes = generate_improve_code(task, programs)
        for source_code in source_codes:
            fn = interpret_code(source_code)
            if fn is not None:
                program = Program(task, source_code, fn)
                try:
                    results = program.create_result_table()
                except Exception:
                    results = "Unable to display evaluation"

                console.print(source_code)
                console.print("\n")
                console.print(results)
                console.print("\n")

                programs.append(program)

                if program.training_success:
                    console.print("Success!!")
                    break

    if file_path:
        console.save_html(file_path)

    return programs


train_set, eval_set = arckit.load_data()

task_id = "f3cdc58f"
task = eval_set[task_id]

programs = solve_task(task, 20, "first_attempt")
