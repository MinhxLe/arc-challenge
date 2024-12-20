"""
refers to all seed datasets
"""

import abc
from dataclasses import dataclass
import functools
import random
from typing import ClassVar, Literal
import arckit
from datasets import Dataset, load_dataset, load_from_disk
import numpy as np
import os
from arc import settings
from arc.external import github, re_arc
from arc.datasets import transform
from loguru import logger

DATA_DIR = os.path.join(settings.TEMP_ROOT_DIR, "data")
GENERATED_DATA_DIR = os.path.join(DATA_DIR, "generated")


class DatasetHandler(abc.ABC):
    def get_dataset(self) -> Dataset:
        return self._create_dataset()

    @abc.abstractmethod
    def _create_dataset(self) -> Dataset:
        """
        returns a dataset of the format
        {
            "train": [
                {"input": list, "output":list }
            ],
            "test": [
                {"input": list, "output":list }
            ]
        }
        """
        pass


@dataclass
class ArckitHandler(DatasetHandler):
    split: Literal["train", "test"]
    version: str = "latest"

    @classmethod
    def _get_arckit_dataset(
        cls, split: Literal["train", "test"], version: str = "latest"
    ) -> arckit.data.TaskSet:
        train_tasks, test_tasks = arckit.load_data(version)
        match split:
            case "train":
                return train_tasks
            case "test":
                return test_tasks
            case _:
                raise ValueError

    @classmethod
    def _format_task(cls, task: arckit.Task) -> dict:
        return dict(
            train=[dict(input=np.array(i), output=np.array(o)) for i, o in task.train],
            test=[dict(input=np.array(i), output=np.array(o)) for i, o in task.test],
        )

    def _create_dataset(self) -> Dataset:
        arckit_dataset = self._get_arckit_dataset(self.split, self.version)
        return Dataset.from_list([self._format_task(t) for t in arckit_dataset])


class ConceptARCHandler(DatasetHandler):
    _REPO_NAME: ClassVar[str] = "victorvikram/ConceptARC"
    _TMP_DIR: ClassVar[str] = os.path.join(DATA_DIR, "concept_arc")

    def __init__(self) -> None:
        if not os.path.exists(self._TMP_DIR):
            os.makedirs(self._TMP_DIR)
            github.clone_repo(self._REPO_NAME, self._TMP_DIR)
        tasks = []

        self.tasks = tasks

    def _create_dataset(self) -> Dataset:
        return load_dataset(
            "json", data_files=f"{self._TMP_DIR}/corpus/*/*.json", split="train"
        )


@dataclass
class GeneratedDatasetHandler(DatasetHandler):
    seed: int

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    def cache_dir(self) -> str:
        return os.path.join(GENERATED_DATA_DIR, self.name)

    def get_dataset(self) -> Dataset:
        if os.path.exists(self.cache_dir):
            logger.info(f"found previously generated dataset at {self.cache_dir}")
            dataset = load_from_disk(self.cache_dir)
        else:
            logger.info(f"creating dataset, saving to {self.cache_dir}")
            dataset = super().get_dataset()
            dataset.save_to_disk(self.cache_dir)
        return dataset


@dataclass
class ReArcHandler(GeneratedDatasetHandler):
    n_tasks: int  # number of tasks to generate per 400
    test_set_size: int = 1
    train_set_size: int = 3

    @property
    def name(self) -> str:
        return f"re_arc_heavy_seed-{self.seed}-train_set_size-{self.train_set_size}_test_set_size-{self.test_set_size}"

    @classmethod
    def _transform_row(cls, row, train_set_size: int, test_set_size: int) -> dict:
        row.pop("task_id")
        examples = row.pop("examples")
        assert len(examples) >= train_set_size + test_set_size
        sampled_examples = random.sample(examples, train_set_size + test_set_size)
        row["train"] = [
            dict(input=i, output=o) for i, o in sampled_examples[:train_set_size]
        ]
        row["test"] = [
            dict(input=i, output=o) for i, o in sampled_examples[train_set_size:]
        ]
        return row

    def _create_dataset(self) -> Dataset:
        random.seed(self.seed)
        # [TODO] we might want to enable
        raw_dataset = re_arc.generate_dataset(
            seed=self.seed,
            n_examples=1000,
            diff_lb=0,
            diff_ub=1,
        )
        repeated_dataset = transform.concat(*[raw_dataset for _ in range(self.n_tasks)])
        return transform.map_dataset(
            repeated_dataset,
            functools.partial(
                self._transform_row,
                train_set_size=self.train_set_size,
                test_set_size=self.test_set_size,
            ),
        )


@dataclass
class ARCHeavyHandler(GeneratedDatasetHandler):
    _HF_NAME = "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems"
    train_set_size: int = 3
    test_set_size: int = 1

    @property
    def name(self) -> str:
        return f"arc_heavy_seed-{self.seed}-train_set_size-{self.train_set_size}_test_set_size-{self.test_set_size}"

    @classmethod
    def _transform_row(cls, row, train_set_size: int, test_set_size: int) -> dict:
        examples = row.pop("examples")
        row.pop("seeds")
        row.pop("source")
        assert len(examples) >= train_set_size + test_set_size
        sampled_examples = random.sample(examples, train_set_size + test_set_size)
        row["train"] = [
            dict(input=i, output=o) for i, o in sampled_examples[:train_set_size]
        ]
        row["test"] = [
            dict(input=i, output=o) for i, o in sampled_examples[train_set_size:]
        ]
        return row

    def _create_dataset(self) -> Dataset:
        random.seed(self.seed)
        raw_dataset = load_dataset(self._HF_NAME, split="train")
        return transform.map_dataset(
            raw_dataset,
            functools.partial(
                self._transform_row,
                train_set_size=self.train_set_size,
                test_set_size=self.test_set_size,
            ),
        )


class Datasets:
    arc_public_train = ArckitHandler("train")
    arc_public_test = ArckitHandler("test")
    concept_arc = ConceptARCHandler()

    create_arc_heavy = ARCHeavyHandler
    create_re_arc = ReArcHandler
