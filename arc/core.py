from dataclasses import dataclass
from enum import IntEnum
import arckit
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

    @classmethod
    def from_dict(cls, example_dict: dict) -> "Example":
        return Example(
            input_=np.array(example_dict["input"]),
            output=np.array(example_dict["output"]),
        )

    def to_dict(self) -> dict:
        return dict(input=self.input_.tolist(), output=self.output.tolist())


@dataclass
class Task:
    id: str | None
    train_set: list[Example]
    test_set: list[Example]

    def to_arckit(self) -> arckit.Task:
        return arckit.Task(
            id=self.id,
            train=[dict(input=x.input_, output=x.output) for x in self.train_set],
            test=[dict(input=x.input_, output=x.output) for x in self.test_set],
        )

    def show(self, show_test_output: bool = False) -> None:
        self.to_arckit().show(answer=show_test_output)

    @classmethod
    def from_dict(cls, task_dict: dict) -> "Task":
        return Task(
            id=task_dict.get("id", None),
            train_set=[Example.from_dict(x) for x in task_dict["train"]],
            test_set=[Example.from_dict(x) for x in task_dict["test"]],
        )

    def to_dict(self) -> dict:
        return dict(
            train=[e.to_dict() for e in self.train_set],
            test=[e.to_dict() for e in self.test_set],
        )


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
