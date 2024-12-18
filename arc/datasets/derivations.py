"""
utils that datasets and generates derived datasets
"""

import multiprocessing
import random
from datasets import Dataset
import numpy as np
from typing import Callable
from arc.datasets.transform import Transform


def map_dataset(dataset: Dataset, fn: Callable) -> Dataset:
    return dataset.map(fn, num_proc=multiprocessing.cpu_count())


def shuffle_train_order(dataset: Dataset, seed: int = 42) -> Dataset:
    random.seed(seed)

    def shuffle(row):
        row["train"] = random.sample(
            row["train"],
            len(row["train"]),
        )

    return map_dataset(dataset, shuffle)


def apply_transform(
    dataset: Dataset,
    transform: Transform,
    transform_input_only: bool = False,
) -> Dataset:
    def apply_transform(row):
        new_train = []
        for example in row["train"]:
            input_ = transform.apply(np.array(example["input"]))
            if transform_input_only:
                output = example["output"]
            else:
                output = transform.apply(np.array(example["output"]))
            new_train.append(dict(input=input_, output=output))
        row["train"] = new_train

        new_test = []
        for example in row["test"]:
            input_ = transform.apply(np.array(example["input"]))
            if transform_input_only:
                output = example["output"]
            else:
                output = transform.apply(np.array(example["output"]))
            new_test.append(dict(input=input_, output=output))
        row["test"] = new_test

    return map_dataset(dataset, apply_transform)
