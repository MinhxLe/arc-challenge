import arckit

from arc.tasks import prompts
import re
from rich.console import Console
from typing import Callable, Optional
import os
from loguru import logger

from arc.external import openai
from arc.program import Program, ProgramExecution

SOLUTION_DIR = "data/openai/code_generation"
os.makedirs(SOLUTION_DIR, exist_ok=True)


def _modified_create_solve_task_prompt(task: arckit.Task) -> str:
    return prompts.create_solve_task_prompt(task) + prompts.addendum_for_nonfinetuned


def _modified_create_improve_solve_task_prompt(
    task: arckit.Task, program_executions: list[ProgramExecution]
) -> str:
    return (
        prompts.create_improve_solve_task_prompt(task, program_executions)
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


def generate_improve_code(
    task, program_executions: list[ProgramExecution]
) -> list[str]:
    print(prompts.programmer_role_prompt)
    print(_modified_create_improve_solve_task_prompt(task, program_executions))
    codes = [
        parse_code(
            openai.complete(
                system_prompt=prompts.programmer_role_prompt,
                prompt=_modified_create_improve_solve_task_prompt(
                    task, program_executions
                ),
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
) -> list[ProgramExecution]:
    program_executions = []
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
        if len(program_executions) == 0:
            source_codes = generate_code(task)
        else:
            source_codes = generate_improve_code(task, program_executions)
        for source_code in source_codes:
            program_execution = ProgramExecution(Program.from_source(source_code), task)

            console.print(source_code)
            console.print("\n")
            console.print(program_execution.create_result_table())
            console.print("\n")
            program_executions.append(program_execution)

            if program_execution.training_success:
                console.print("Success!!")
                break

    if file_path:
        console.save_html(file_path)

    return program_executions
