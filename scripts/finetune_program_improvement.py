from arckit.data import Task
from arc.datasets.barc_modified_programs import get_raw_dataset
from arc.types import Program
from dataclasses import dataclass
import numpy as np
from concurrent.futures import Future, ProcessPoolExecutor
import multiprocessing
from typing import Tuple, Dict
from loguru import logger


dataset = get_raw_dataset()["train"]


@dataclass
class EvaluationSummary:
    train_correct: int
    train_total: int


def evaluate_program(program: Program, task: Task, i: int) -> EvaluationSummary:
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


if __name__ == "__main__":
    validate_dataset()
