import typing as ta
from enum import IntEnum, Enum
import numpy as np


Grid = np.ndarray[ta.Tuple[int, int], np.dtype[np.integer]]


class Color(IntEnum):
    BLACK = 0
    BLUE = 1
    RED = 2
    YELLOW = 3
    PURPLE = 4
    ORANGE = 5
    GREEN = 6
    BROWN = 7
    GREY = 8
    PINK = 9


class Concept(str, Enum):
    COLOR_MAPPING = "color mapping"
    REFLECTION = "reflection"
    ROTATION = "rotation"
    SYMMETRY = "symmetry"
    TRANSLATION = "translation"
