import re
import tqdm
import asyncio
from dataclasses import dataclass
from typing import Any
from arc.datasets.barc_generated_problems import get_parsed_dataset, GeneratedTask
from arc.external import openai
from arc.program import Program
from arc.tasks import prompts
import random
import itertools
from arc import utils


async def generate_incorrect_programs(task: GeneratedTask, difficulty: int):
    assert 1 <= difficulty <= 10
    return await openai.complete_async(
        system_prompt=f"""{prompts.programmer_role_prompt}
You are given a problem specification and the source code for a solution for the program. Your intention is to modify the source code such that the program no longer satisfy the specification. The changed program is intended to be given to a student programmer so that they will be able to revert it. Given on a difficulty score of 1 to 10, you should output the modified program such that a program of score 1 will be easy to debug and 10 will most challenging/requires most change.

Changes should
1. change the program behavior.
2. NOT be variable renaming.
3. NOT change the output type.
4. NOT contain comments highlighting the introduced bug.
""",
        prompt=f"""Description:
{task.description}

Original source code:
```python
{task.program.source}

Output a modified program to debug with a difficulty of {difficulty}.
```""",
        return_raw=True,
    )


# batch iterate through tasks of size 10
SAMPLE_SIZE = 1000
BATCH_SIZE = 20


@dataclass
class IncorrectProgram:
    task: GeneratedTask
    difficulty: int
    response: Any


async def generate_incorrect_program_datasets(
    num_tasks: int,
    difficulties: list[int],
) -> list[IncorrectProgram]:
    batch_size = 20
    assert len(difficulties) > 0, "provide at least 1 difficulty"
    tasks = get_parsed_dataset()
    sampled_tasks = random.sample(tasks, k=num_tasks)

    incorrect_programs = []
    for batch in tqdm.tqdm(
        utils.batch(itertools.product(sampled_tasks, difficulties), batch_size)
    ):
        responses = await asyncio.gather(
            *[
                generate_incorrect_programs(task, difficulty)
                for task, difficulty in batch
            ]
        )
        batch_programs = [
            IncorrectProgram(task, difficulty, response)
            for (task, difficulty), response in zip(batch, responses)
        ]
        incorrect_programs.extend(batch_programs)
    return incorrect_programs


def full_cost(programs: list[IncorrectProgram]):
    sum(openai.calculate_cost(program.response.response) for program in programs)


def extract_program(msg: str) -> Program:
    # Extract code between Python code blocks
    code_match = re.search(r"```python\n(.*?)```", msg, re.DOTALL)
    assert code_match is not None
    source_code = code_match.group(1)
    return Program.from_source(source_code)
