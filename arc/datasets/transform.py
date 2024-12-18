import abc
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

from arc.core import Color, Grid


class Transform(abc.ABC):
    @abc.abstractmethod
    def apply(self, grid: Grid) -> Grid:
        pass

    @abc.abstractmethod
    def inverse_apply(self, grid: Grid) -> Grid:
        pass


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

    def inverse_apply(self, grid: Grid) -> Grid:
        return np.rot90(grid, k=-self.k)


@dataclass
class Reflect(Transform):
    class Type(Enum):
        VERTICAL = auto()
        HORIZONTAL = auto()

    type_: Type

    def apply(self, grid: Grid) -> Grid:
        match self.type_:
            case self.Type.HORIZONTAL:
                return np.flip(grid, 0)
            case self.Type.VERTICAL:
                return np.flip(grid, 1)

    def inverse_apply(self, grid: Grid) -> Grid:
        return self.apply(grid)


@dataclass
class MapColor(Transform):
    mapping: dict[Color, Color]

    def __post_init__(self):
        # we don't want to allow mapping background
        assert Color.BLACK not in self.mapping.keys()
        assert Color.BLACK not in self.mapping.values()
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}

    def apply(self, grid: Grid) -> Grid:
        return self._apply_mapping(grid, self.mapping)

    def inverse_apply(self, grid: Grid) -> Grid:
        return self._apply_mapping(grid, self.inverse_mapping)

    @classmethod
    def _apply_mapping(cls, grid: Grid, mapping: dict[Color, Color]) -> Grid:
        return np.vectorize(lambda x: mapping.get(x, x))(grid)


@dataclass
class Compose(Transform):
    transforms: list[Transform]

    def apply(self, grid: Grid) -> Grid:
        for t in self.transforms:
            grid = t.apply(grid)
        return grid

    def inverse_apply(self, grid: Grid) -> Grid:
        # [IMPORANT] this is not 100% correct because this assumes that transforms
        # are commutative and D8 is not a commutative group.
        for t in reversed(self.transforms):
            grid = t.inverse_apply(grid)
        return grid
