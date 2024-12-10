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


def evaluate_program(program: Program, task: Task) -> EvaluationSummary:
    train_correct = 0
    for input_, output in task.train:
        program_output = program.call(input_)
        break
        if not isinstance(program_output, Exception):
            if program_output.shape == output.shape and np.all(
                program_output == output
            ):
                train_correct += 1
    return EvaluationSummary(train_correct=train_correct, train_total=len(task.train))


# Some data  validation
def process_row(row_data: Tuple[int, Dict]) -> int:
    i, row = row_data
    task = Task(**row["task"])
    original_program = Program.from_source(row["original_program_source"])
    modified_program = Program.from_source(row["modified_program_source"])

    # ensuring programs changed
    if original_program.source == modified_program.source:
        return -1  # Skip this one

    # ensuring original program is right
    original_evaluation = evaluate_program(original_program, task)
    if original_evaluation.train_total != original_evaluation.train_correct:
        return i
    return -1


def main():
    # Get the number of CPU cores, leave one free for system
    num_processes = max(1, multiprocessing.cpu_count() - 1)

    # Create enumerated dataset for processing
    enumerated_dataset = list(enumerate(dataset))[:10]

    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures: list[Future] = [
            executor.submit(process_row, r) for r in enumerated_dataset
        ]
    for future in futures:
        try:
            print(future.result())
        except Exception as e:
            logger.exception(e)


if __name__ == "__main__":
    main()

# 838
# 839
# 958
# 959
