import os
from dataclasses import dataclass
from enum import StrEnum
import abc
from typing import Callable, ClassVar
import arckit
from datasets.load import load_dataset
from arc.core import Example, Task
from loguru import logger
from arc import settings
import json
import re
import numpy as np
import glob
import random
import zipfile
from arc.external import github

TMP_DATA_DIR = os.path.join(settings.TEMP_ROOT_DIR, "data")
NEONEYE_REPO_NAME = "neoneye/arc-dataset-collection"
NEONEYE_DATA_DIR = os.path.join(TMP_DATA_DIR, "neoneye_arc_dataset_collection")


class Split(StrEnum):
    TEST = "test"
    TRAIN = "train"


def _get_tasks_from_json_dir(
    dir: str,
    fname_to_id: Callable[[str], str | None] | None,
) -> list[Task]:
    tasks = []
    # Find all JSON files in the concept directory
    for json_file in os.listdir(dir):
        if not json_file.endswith(".json"):
            continue
        if fname_to_id is not None:
            id = fname_to_id(json_file)
        else:
            id = None
        json_path = os.path.join(dir, json_file)
        try:
            with open(json_path, "r") as f:
                task_data = json.load(f)
        except Exception as e:
            logger.warning(f"Error processing {json_path}: {str(e)}")
            continue

        def _to_example(example_dict: dict) -> Example:
            return Example(
                input_=np.array(example_dict["input"]),
                output=np.array(example_dict["output"]),
            )

        train_set = list(map(_to_example, task_data["train"]))
        test_set = list(map(_to_example, task_data["test"]))
        task = Task(id=id, train_set=train_set, test_set=test_set)
        tasks.append(task)
    return tasks


class ArcDataset(abc.ABC):
    """
    an ARC dataset is a common interface into working with arc tasks.
    [TODO] add train/test split support
    """

    @abc.abstractmethod
    def get_tasks(self) -> list[Task]: ...

    def to_hf_dataset(self):
        pass

    def shuffle_train_order(self, seed: int = 42) -> "SimpleDataset":
        random.seed(seed)
        shuffled_tasks = []
        for task in self.get_tasks():
            shuffled_tasks.append(
                Task(
                    id=task.id,
                    train_set=random.sample(task.train_set, len(task.train_set)),
                    test_set=task.test_set,
                )
            )
        return SimpleDataset(shuffled_tasks)

    def from_train_set(self) -> "SimpleDataset":
        raise NotImplementedError
        pass


@dataclass
class SimpleDataset(ArcDataset):
    tasks: list[Task]

    def get_tasks(self) -> list[Task]:
        return self.tasks


@dataclass
class ArckitDataset(ArcDataset):
    split: Split
    version: str = "latest"

    def __post_init__(self) -> None:
        train_tasks, test_tasks = arckit.load_data(self.version)
        match self.split:
            case Split.TEST:
                self.tasks = [self._from_arkit_task(t) for t in train_tasks]
            case Split.TRAIN:
                self.tasks = [self._from_arkit_task(t) for t in test_tasks]
            case _:
                raise ValueError

    def get_tasks(self) -> list[Task]:
        return self.tasks

    def _from_arkit_task(self, arckit_task: arckit.Task) -> Task:
        train_set = [Example(i, o) for i, o in arckit_task.train]
        test_set = [Example(i, o) for i, o in arckit_task.test]
        return Task(arckit_task.id, train_set, test_set)


class ConceptARCDataset(ArcDataset):
    _REPO_NAME: ClassVar[str] = "victorvikram/ConceptARC"
    _TMP_DIR: ClassVar[str] = os.path.join(TMP_DATA_DIR, "concept_arc")

    def __post_init__(self) -> None:
        if not os.path.exists(self._TMP_DIR):
            github.clone(self._REPO_NAME)
        tasks = []
        corpus_dir = os.path.join(self._TMP_DIR, "corpus")

        # Get all directories in corpus
        for concept_dir in os.listdir(corpus_dir):
            concept_path = os.path.join(corpus_dir, concept_dir)
            if not os.path.isdir(concept_path):
                continue

            def fname_to_id(fname: str) -> str | None:
                match = re.search(r".*(\d+)\.json$", fname)
                if not match:
                    return None
                suffix_number = int(match.group(1))
                return f"{concept_dir}_{suffix_number}"

            tasks.extend(_get_tasks_from_json_dir(concept_path, fname_to_id))

        self.tasks = tasks

    @classmethod
    def _get_task_id(cls, concept: str, num: int) -> str:
        return f"concept_arc_{concept}_{num}"

    @classmethod
    def _to_example(cls, example_dict: dict) -> Example:
        return Example(
            input_=np.array(example_dict["input"]),
            output=np.array(example_dict["output"]),
        )

    def get_tasks(self) -> list[Task]:
        return self.tasks


@dataclass
class ArcHeavyDataset(ArcDataset):
    _TMP_DIR = os.path.join(TMP_DATA_DIR, "arc_heavy")

    def __init__(self) -> None:
        if not os.path.exists(NEONEYE_DATA_DIR):
            github.clone(NEONEYE_REPO_NAME, NEONEYE_DATA_DIR)

        if not os.path.exists(self._TMP_DIR):
            logger.debug(f"creating {self._TMP_DIR}")
            os.makedirs(self._TMP_DIR)
            heavy_data_dir = os.path.join(
                NEONEYE_DATA_DIR, "dataset", "ARC-Heavy", "data_100k"
            )
            zip_files = glob.glob(os.path.join(heavy_data_dir, "*.zip"))

            for zip_path in zip_files:
                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        # Extract to a subdirectory named after the zip file (without .zip extension)
                        zip_ref.extractall(self._TMP_DIR)
                        logger.debug(f"Extracted {zip_path}")
                except Exception as e:
                    logger.exception(f"Failed to extract {zip_path}: {str(e)}")
                    continue

        def fname_to_id(fname: str) -> str | None:
            match = re.search(r"task(\d+)", fname)
            if match is not None:
                id = f"arc_heavy_{match.group(1)}"
            else:
                id = None
            return id

        # this is too slow
        # self.tasks = _get_tasks_from_json_dir(self._TMP_DIR, fname_to_id)
        load_dataset("json", self._TMP_DIR, "*.json", num_proc=8)["train"]

    def get_tasks(self) -> list[Task]:
        return []
