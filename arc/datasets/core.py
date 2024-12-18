from dataclasses import dataclass
from enum import StrEnum
import abc
import arckit
from arc.core import Example, Task
from loguru import logger


class Split(StrEnum):
    TEST = "test"
    TRAIN = "train"


class ArcDataset(abc.ABC):
    """
    an ARC dataset is a common interface into working with arc tasks.
    [TODO] add train/test split support
    """

    @abc.abstractmethod
    def get_tasks(self, split: Split) -> list[Task]: ...

    def to_hf_dataset(self):
        pass


@dataclass
class ArckitDataset(ArcDataset):
    version: str = "latest"

    def __post_init__(self) -> None:
        train_tasks, test_tasks = arckit.load_data(self.version)

        self.train_tasks = [self._from_arkit_task(t) for t in train_tasks]
        self.test_tasks = [self._from_arkit_task(t) for t in test_tasks]

    def get_tasks(self, split: Split) -> list[Task]:
        match split:
            case Split.TEST:
                return self.test_tasks
            case Split.TRAIN:
                return self.train_tasks

    def _from_arkit_task(self, arckit_task: arckit.Task) -> Task:
        train_set = [Example(i, o) for i, o in arckit_task.train]
        if not len(arckit_task.test) == 1:
            logger.warning(f"arckit task {arckit_task.id} has more than 1 test")
        test = [Example(i, o) for i, o in arckit_task.test][0]
        return Task(arckit_task.id, train_set, test)
