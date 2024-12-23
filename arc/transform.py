import abc
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

from arc.core import Color, Grid


class Transform(abc.ABC):
    @abc.abstractmethod
    def apply(self, grid: Grid) -> Grid:
        pass

    @property
    @abc.abstractmethod
    def inverse(self) -> "Transform":
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

    @property
    def inverse(self) -> "Rotate":
        return Rotate(k=-self.k)


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

    @property
    def inverse(self) -> "Reflect":
        return self


@dataclass
class MapColor(Transform):
    mapping: dict[Color, Color]

    def __post_init__(self):
        # we don't want to allow mapping background
        assert Color.BLACK not in self.mapping.keys()
        assert Color.BLACK not in self.mapping.values()
        assert len(self.mapping) == len(set(self.mapping.values()))

    def apply(self, grid: Grid) -> Grid:
        return self._apply_mapping(grid, self.mapping)

    @property
    def inverse(self) -> "MapColor":
        return MapColor({v: k for v, k in self.mapping.items()})

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

    @property
    def inverse(self) -> "Compose":
        # [IMPORANT] this is not 100% correct because this assumes that transforms
        # are commutative and D8 is not a commutative group.
        return Compose(list(reversed(self.transforms)))
