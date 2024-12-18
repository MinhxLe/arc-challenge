from dataclasses import dataclass
from enum import IntEnum
import numpy as np

MIN_GRID_WIDTH = 1
MAX_GRID_WIDTH = 30
MIN_GRID_HEIGHT = 1
MAX_GRID_HEIGHT = 30

# Tuple of width and height and each entry is an integer representing the color.
Grid = np.ndarray  # [ta.Tuple[int, int], np.dtype[np.integer]]


@dataclass
class Example:
    input_: Grid
    output: Grid


@dataclass
class Task:
    id: str | None
    train_set: list[Example]
    test: Example

    def to_arckit(self):
        raise NotImplementedError


class Color(IntEnum):
    """
    Colors are strings (NOT integers), so you CAN'T do math/arithmetic/indexing/ordering on them.
    """

    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    GRAY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9

    # why tho
    PURPLE = 8
    BROWN = 9

    # Keep these below BLACK so that Color(0).name returns 'BLACK'
    TRANSPARENT = 0  # sometimes the language model likes to pretend that there is something called transparent/background, and black is a reasonable default
    BACKGROUND = 0

    @classmethod
    @property
    def ALL_COLORS(cls) -> list["Color"]:
        return [c for c in cls]

    @classmethod
    @property
    def NOT_BLACK(cls) -> list["Color"]:
        return [c for c in cls if c != Color.BLACK]
