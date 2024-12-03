import re
import asyncio
from dataclasses import dataclass

from datasets import Dataset, DatasetDict
from arc.datasets.barc_generated_problems import get_parsed_dataset, GeneratedTask
from arc.external import openai
from arc.tasks import prompts
import random
import itertools
from arc import utils
from openai import RateLimitError
from loguru import logger
import pandas as pd
from arc.external import huggingface

SAMPLE_SIZE = 1000
DIFFULTIES = [1, 10]


@utils.async_retry(delay=30, exceptions=(RateLimitError))
async def generate_incorrect_program(task: GeneratedTask, difficulty: int):
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


@dataclass
class Row:
    generated_task: GeneratedTask
    difficulty: int
    response: openai.RawCompletion[str]


async def generate_modified_programs(
    tasks: list[GeneratedTask],
    difficulties: list[int],
) -> list[Row]:
    batch_size = 20
    assert len(difficulties) > 0, "provide at least 1 difficulty"

    incorrect_programs = []
    num_batches = len(tasks) * len(difficulties) // batch_size
    try:
        for i, batch in enumerate(
            utils.batch(itertools.product(tasks, difficulties), batch_size)
        ):
            logger.info(f"processing {i}/{num_batches} batches")
            responses = await asyncio.gather(
                *[
                    generate_incorrect_program(task, difficulty)
                    for task, difficulty in batch
                ]
            )
            batch_programs = [
                Row(task, difficulty, response)
                for (task, difficulty), response in zip(batch, responses)
            ]
            incorrect_programs.extend(batch_programs)
    except KeyboardInterrupt:
        logger.warning("ending prematurely due to keyboard interreupt")
    return incorrect_programs


def extract_source(msg: str) -> str:
    # Extract code between Python code blocks
    code_match = re.search(r"```python\n(.*?)```", msg, re.DOTALL)
    assert code_match is not None
    return code_match.group(1)


def row_to_dict(row: Row) -> dict:
    task = row.generated_task.task

    return dict(
        task_description=row.generated_task.description,
        task=task.to_dict(),
        difficulty=row.difficulty,
        original_program_source=row.generated_task.program.source,
        modified_program_source=extract_source(row.response.completion),
        raw_llm_response=row.response.completion,
    )


async def main():
    tasks = get_parsed_dataset()
    random.shuffle(tasks)
    sampled_tasks = tasks[:SAMPLE_SIZE]
    rows = await generate_modified_programs(sampled_tasks, DIFFULTIES)
    data = [row_to_dict(r) for r in rows]
    df = pd.DataFrame(data)
    # there is no test here?
    dataset = DatasetDict(dict(train=Dataset.from_pandas(df)))

    huggingface.upload_dataset(
        "minhxle/barc-induction-modified-programs-2k",
        dataset,
    )

    # TODO validate the modified program actually changes train or test input/output pair
