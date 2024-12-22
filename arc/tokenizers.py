"""
Custom serialization for tasks
"""

from dataclasses import dataclass
from arc.core import Task, Grid, Color
import numpy as np
from typing import TypedDict


class Token:
    NEW_LINE = "\n"
    INPUT = "I"
    OUTPUT = "O"


class SFTRow(TypedDict):
    train: str
    query: str
    reply: str
    text: str


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
            serialized += self._format_output(output_grid)
        return serialized

    def _format_output(self, grid: Grid) -> str:
        return f"{self.format_grid(grid)}{self.output_tail_token}"

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
    def format_task_for_sft(self, task: Task) -> SFTRow:
        train_text = "".join(
            [self._format_input_output(x.input_, x.output) for x in task.train_set]
        )
        train_text = self.preprompt + train_text
        test_example = task.test_set[0]
        query_text = self._format_input_output(test_example.input_, None)
        reply_text = self._format_output(test_example.output)
        return SFTRow(
            train=train_text,
            query=query_text,
            reply=reply_text,
            text=f"{train_text}{query_text}{reply_text}",
        )

    def transform_train_test_to_text_schema(self, row: dict) -> dict:
        task = Task.from_dict(row)
        row.pop("train")
        row.pop("test")
        row["text"] = self.format_task(task, include_test_output=True)
        return row
