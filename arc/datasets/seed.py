"""
refers to all seed datasets
"""

import abc
from dataclasses import dataclass
from typing import ClassVar, Literal
import arckit
from datasets import Dataset, load_dataset
import numpy as np
import os
from arc import settings
from arc.external import github
from loguru import logger
import glob
import zipfile
import multiprocessing

TMP_DATA_DIR = os.path.join(settings.TEMP_ROOT_DIR, "data")
_NEONEYE_REPO_NAME = "neoneye/arc-dataset-collection"
_NEONEYE_DATA_DIR = os.path.join(TMP_DATA_DIR, "neoneye_arc_dataset_collection")


class DatasetHandler(abc.ABC):
    @abc.abstractmethod
    def get_dataset(self) -> Dataset:
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

    def get_dataset(self) -> Dataset:
        arckit_dataset = self._get_arckit_dataset(self.split, self.version)
        return Dataset.from_list([self._format_task(t) for t in arckit_dataset])


class ConceptARCHandler(DatasetHandler):
    _REPO_NAME: ClassVar[str] = "victorvikram/ConceptARC"
    _TMP_DIR: ClassVar[str] = os.path.join(TMP_DATA_DIR, "concept_arc")

    def __post_init__(self) -> None:
        if not os.path.exists(self._TMP_DIR):
            github.clone_repo(self._REPO_NAME, self._TMP_DIR)
        tasks = []

        self.tasks = tasks

    def get_dataset(self) -> Dataset:
        return load_dataset(
            "json", data_files=f"{self._TMP_DIR}/corpus/*/*.json", split="train"
        )


class ARCHeavyHandler(DatasetHandler):
    _TMP_DIR = os.path.join(TMP_DATA_DIR, "arc_heavy")

    def __init__(self) -> None:
        if not os.path.exists(_NEONEYE_DATA_DIR):
            github.clone_repo(_NEONEYE_REPO_NAME, _NEONEYE_DATA_DIR)

        if not os.path.exists(self._TMP_DIR):
            logger.debug(f"creating {self._TMP_DIR}")
            os.makedirs(self._TMP_DIR)
            heavy_data_dir = os.path.join(
                _NEONEYE_DATA_DIR, "dataset", "ARC-Heavy", "data_100k"
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
        # TODO it's worth optimizing merging in the datasets into a single file

    def get_dataset(self) -> Dataset:
        # self.tasks = _get_tasks_from_json_dir(self._TMP_DIR, fname_to_id)
        return load_dataset(
            "json",
            data_files=f"{self._TMP_DIR}/*.json",
            num_proc=multiprocessing.cpu_count(),
            split="train",
        )


class Datasets:
    arc_public_train = ArckitHandler("train")
    arc_public_test = ArckitHandler("test")
    concept_arc = ConceptARCHandler()
    arc_heavy = ARCHeavyHandler()  # [TODO][P0] I think this is not correct
