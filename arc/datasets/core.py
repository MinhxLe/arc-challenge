import os
from dataclasses import dataclass
from enum import StrEnum
import abc
import arckit
from arc.core import Example, Task
from loguru import logger
from arc import settings
import subprocess
import json
import re
import numpy as np

TMP_DATA_DIR = os.path.join(settings.TEMP_ROOT_DIR, "data")


class Split(StrEnum):
    TEST = "test"
    TRAIN = "train"


class ArcDataset(abc.ABC):
    """
    an ARC dataset is a common interface into working with arc tasks.
    [TODO] add train/test split support
    """

    @abc.abstractmethod
    def get_tasks(self) -> list[Task]: ...

    def to_hf_dataset(self):
        pass


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


@dataclass
class ConceptARCDataset(ArcDataset):
    _REPO_URL = "https://github.com/victorvikram/ConceptARC"
    _TMP_DIR = os.path.join(TMP_DATA_DIR, "concept_arc")

    def __post_init__(self) -> None:
        if not os.path.exists(self._TMP_DIR):
            logger.debug("cloning ConceptARC dataset")
            subprocess.run(["git", "clone", self._REPO_URL, self._TMP_DIR], check=True)
        tasks = []
        corpus_dir = os.path.join(self._TMP_DIR, "corpus")

        # Get all directories in corpus
        for concept_dir in os.listdir(corpus_dir):
            concept_path = os.path.join(corpus_dir, concept_dir)
            if not os.path.isdir(concept_path):
                continue

            # Find all JSON files in the concept directory
            for json_file in os.listdir(concept_path):
                if not json_file.endswith(".json"):
                    continue

                # Extract the suffix number using regex
                match = re.search(r".*(\d+)\.json$", json_file)
                if not match:
                    continue

                suffix_number = int(match.group(1))
                json_path = os.path.join(concept_path, json_file)

                try:
                    with open(json_path, "r") as f:
                        task_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Error processing {json_path}: {str(e)}")
                    continue

                # Create task ID from directory name and suffix
                task_id = f"{concept_dir}_{suffix_number}"

                # Convert the JSON data into Examples
                train_set = list(map(self._to_example, task_data["train"]))
                test_set = list(map(self._to_example, task_data["test"]))
                task = Task(id=task_id, train_set=train_set, test_set=test_set)
                tasks.append(task)
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
