import typing as ta
from enum import IntEnum


Grid = ta.Tuple[ta.Tuple[int]]


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
