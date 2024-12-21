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
    # this should be parametrized based on model's EOS tokenizer
    output_tail_token: str
    # [TODO][QUESTION] why do we need this?
    preprompt: str = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz"
    input_head_token: str = Token.INPUT
    output_head_token: str = Token.OUTPUT

    def format_grid(self, grid: Grid) -> str:
        min_color, max_color = min(Color).value, max(Color).value
        assert (grid >= min_color).all()
        assert (grid <= max_color).all()

        char_grid = np.char.mod("%d", grid).tolist()
        return Token.NEW_LINE.join(["".join(row) for row in char_grid]) + Token.NEW_LINE

    def _format_input_output(self, input_grid: Grid, output_grid: Grid | None) -> str:
        serialized = f"{self.input_head_token}{self.format_grid(input_grid)}{self.output_head_token}"
        if output_grid is not None:
            serialized += f"{self.format_grid(output_grid)}{self.output_tail_token}"
        return serialized

    def format_task(self, task: Task, include_test_output: bool) -> str:
        train_text = "".join(
            [self._format_input_output(x.input_, x.output) for x in task.train_set]
        )
        test_example = task.test_set[0]
        test_text = self._format_input_output(
            test_example.input_, test_example.output if include_test_output else None
        )

        return f"{self.preprompt}{train_text}{test_text}"

    # TODO: type the row and return more tightly
    def transform_train_test_to_text_schema(self, row: dict) -> dict:
        task = Task.from_dict(row)
        row.pop("train")
        row.pop("test")
        row["text"] = self.format_task(task, include_test_output=True)
        return row
