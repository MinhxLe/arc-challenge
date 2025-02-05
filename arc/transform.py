import abc
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
import random

from arc.core import Color, Grid, Task


class Transform(abc.ABC):
    @abc.abstractmethod
    def apply(self, grid: Grid) -> Grid:
        pass

    @property
    @abc.abstractmethod
    def inverse(self) -> "Transform":
        pass


@dataclass
class Identity(Transform):
    def apply(self, grid: Grid) -> Grid:
        return grid

    @property
    def inverse(self) -> "Transform":
        return Identity()


@dataclass
class Rotate(Transform):
    """
    90 degree counterclockwise rotation
    """

    k: int

    def __post_init__(self):
        assert -3 <= self.k <= 3

    def apply(self, grid: Grid) -> Grid:
        return np.rot90(grid, k=self.k)

    @property
    def inverse(self) -> "Rotate":
        return Rotate(k=-self.k)


@dataclass
class Reflect(Transform):
    class Type(Enum):
        VERTICAL = auto()
        HORIZONTAL = auto()
        DIAGONAL = auto()
        OTHER_DIAGONAL = auto()

    type_: Type

    def apply(self, grid: Grid) -> Grid:
        match self.type_:
            case self.Type.HORIZONTAL:
                return np.flip(grid, 0)
            case self.Type.VERTICAL:
                return np.flip(grid, 1)
            case self.Type.DIAGONAL:
                return np.swapaxes(grid, 0, 1)
            case self.Type.OTHER_DIAGONAL:
                return np.rot90(np.swapaxes(grid, 0, 1), 2)

    @property
    def inverse(self) -> "Reflect":
        return self


class MapColor(Transform):
    def __init__(self, mapping: dict[Color, Color]):
        assert len(mapping) == len(set(mapping.values()))
        self.mapping = mapping

    def apply(self, grid: Grid) -> Grid:
        return self._apply_mapping(grid, self.mapping)

    @property
    def inverse(self) -> "MapColor":
        return MapColor({v: k for v, k in self.mapping.items()})

    @classmethod
    def _apply_mapping(cls, grid: Grid, mapping: dict[Color, Color]) -> Grid:
        return np.vectorize(lambda x: mapping.get(x, x))(grid)


@dataclass
class PermuteColor(MapColor):
    def __init__(self, seed: int):
        random.seed(seed)
        self.seed = seed
        shuffled_colors = [c for c in Color]
        random.shuffle(shuffled_colors)
        mapping = {
            original: new for original, new in zip([c for c in Color], shuffled_colors)
        }
        super().__init__(mapping)


@dataclass
class Compose(Transform):
    transforms: list[Transform]

    def apply(self, grid: Grid) -> Grid:
        for t in self.transforms:
            grid = t.apply(grid)
        return grid

    @property
    def inverse(self) -> "Compose":
        # [IMPORANT] this is not 100% correct because this assumes that transforms
        # are commutative and D8 is not a commutative group.
        return Compose(list(reversed(self.transforms)))


def create_random_transform(seed: int) -> Transform:
    random.seed(seed)
    return Compose(
        transforms=[
            Rotate(k=random.randint(0, 3)),
            Reflect(type_=random.choice(list(Reflect.Type))),
            PermuteColor(seed=seed),
        ]
    )


def generate_train_only_tasks(task: Task) -> list[Task]:
    if len(task.train_set) <= 1:
        return []

    new_train = task.train_set.copy()
    new_test = [new_train.pop(-1)]

    return [Task(id=None, train_set=new_train, test_set=new_test)]
