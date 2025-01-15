"""
utils that datasets and generates derived datasets
"""

import multiprocessing
import random
from datasets import Dataset
import numpy as np
from typing import Callable
from arc.transform import Transform
from datasets import concatenate_datasets


def sample(dataset: Dataset, n: int, seed: int = 42) -> Dataset:
    assert len(dataset) >= n
    return dataset.shuffle(seed=seed).select(range(n))


def concat(*datasets: Dataset) -> Dataset:
    return concatenate_datasets(datasets)


def repeat(dataset: Dataset, n: int) -> Dataset:
    return concat(*[dataset for _ in range(n)])


def map_dataset(dataset: Dataset, fn: Callable) -> Dataset:
    return dataset.map(fn, num_proc=multiprocessing.cpu_count())


def shuffle_train_order(dataset: Dataset, seed: int) -> Dataset:
    random.seed(seed)

    def shuffle(row):
        random.shuffle(row["train"])
        return row

    return map_dataset(dataset, shuffle)


def apply_transform(
    dataset: Dataset,
    transform: Transform,
    input_only: bool = False,
) -> Dataset:
    def apply_transform(row):
        new_train = []
        for example in row["train"]:
            input_ = transform.apply(np.array(example["input"]))
            if input_only:
                output = example["output"]
            else:
                output = transform.apply(np.array(example["output"]))
            new_train.append(dict(input=input_, output=output))
        row["train"] = new_train
        new_test = []
        for example in row["test"]:
            input_ = transform.apply(np.array(example["input"]))
            if input_only:
                output = example["output"]
            else:
                output = transform.apply(np.array(example["output"]))
            new_test.append(dict(input=input_, output=output))
        row["test"] = new_test
        return row

    return map_dataset(dataset, apply_transform)


def generate_ttt_dataset(dataset: Dataset) -> Dataset:
    def create_ttt_tasks(row):
        if len(row["train"]) <= 1:
            return []

        ttt_tasks = []
        # TODO: Should we generate only one new task to avoid the test
        # showing up in train context in other permutations?
        # TODO: Should we shuffle train order?
        for idx in range(len(row["train"])):
            new_task_train_copy = row["train"].copy()
            new_test = [new_task_train_copy.pop(idx)]
            ttt_tasks.append({"train": new_task_train_copy, "test": new_test})

        return {"tasks": ttt_tasks}

    return Dataset.from_list(
        [
            task
            for ttt_tasks in map_dataset(dataset, create_ttt_tasks)["tasks"]
            for task in ttt_tasks
        ]
    )
