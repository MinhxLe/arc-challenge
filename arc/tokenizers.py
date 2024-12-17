"""
Custom serialization for tasks
"""

from dataclasses import dataclass
from arc.core import Task, Grid, Color
import numpy as np


class Token:
    NEW_LINE = "\n"
    INPUT = "I"
    OUTPUT = "O"


@dataclass
class Formatter:
    # this should be parametrized based on model's tokenizer
    bos_token: str
    eos_token: str
    pad_token: str
    # [TODO][QUESTION] why do we need this?
    preprompt: str = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz"
    input_header_token: str = Token.INPUT
    output_header_token: str = Token.OUTPUT

    def format_grid(self, grid: Grid) -> str:
        min_color, max_color = min(Color).value, max(Color).value
        assert (grid >= min_color).all()
        assert (grid < max_color).all()

        char_grid = np.char.mod("%d", grid).tolist()
        return Token.NEW_LINE.join(["".join(row) for row in char_grid]) + Token.NEW_LINE

    def _format_input_output(self, input_grid: Grid, output_grid: Grid | None) -> str:
        serialized = f"{self.input_header_token}{self.format_grid(input_grid)}{self.output_header_token}"
        if output_grid is not None:
            serialized += f"{self.format_grid(output_grid)}{self.eos_token}"
        return serialized

    def format_task(self, task: Task, include_test_output: bool) -> str:
        train_text = "".join(
            [self._format_input_output(x.input_, x.output) for x in task.train_set]
        )
        test_text = self._format_input_output(
            task.test.input_, task.test.output if include_test_output else None
        )

        return f"{self.preprompt}{train_text}{test_text}"
