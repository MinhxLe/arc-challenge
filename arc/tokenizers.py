"""
Custom serialization for tasks
"""

from dataclasses import dataclass
from arc.core import Task, Grid, Color, Example
import numpy as np
from typing import Tuple, TypedDict


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

    def _validate_grid(self, grid: Grid) -> None:
        min_color, max_color = min(Color).value, max(Color).value
        assert (grid >= min_color).all()
        assert (grid <= max_color).all()

    def format_grid(self, grid: Grid) -> str:
        self._validate_grid(grid)
        char_grid = np.char.mod("%d", grid).tolist()
        return Token.NEW_LINE.join(["".join(row) for row in char_grid]) + Token.NEW_LINE

    def parse_grid(self, raw: str) -> Grid:
        """Parse a serialized grid string back into a Grid (numpy array).

        Args:
            raw: String representation of grid, with rows separated by newlines
                 and single digit numbers representing colors

        Returns:
            Grid: numpy array containing the parsed color values
        """
        raw = raw.rstrip()
        rows = [row for row in raw.split(Token.NEW_LINE)]
        grid_list = [[int(cell) for cell in row] for row in rows]
        grid = np.array(grid_list)
        self._validate_grid(grid)
        return grid

    def _format_input_output(self, input_grid: Grid, output_grid: Grid | None) -> str:
        serialized = f"{self.input_head_token}{self.format_grid(input_grid)}{self.output_head_token}"
        if output_grid is not None:
            serialized += self._format_output(output_grid)
        return serialized

    def _parse_input_output(self, raw: str) -> Tuple[Grid, Grid | None]:
        """Parse a string containing an input grid and optional output grid.

        Args:
            raw: String starting with 'I' followed by input grid, optionally followed
                 by 'O' and output grid

        Returns:
            Tuple of (input Grid, output Grid or None)
        """
        # Split on output token if present
        parts = raw.split(self.output_head_token)

        if len(parts) != 2:
            raise ValueError(
                f"missing output head token {self.output_head_token} in {raw}"
            )

        if not parts[0].startswith(self.input_head_token):
            raise ValueError(f"Input section must start with {self.input_head_token}")
        input_grid = self.parse_grid(parts[0][len(self.input_head_token) :])

        # Parse output if present
        output_str = parts[1]
        if output_str == "":
            output_grid = None
        else:
            output_grid = self._parse_output(output_str)
        return input_grid, output_grid

    def _format_output(self, grid: Grid) -> str:
        return f"{self.format_grid(grid)}{self.output_tail_token}"

    def _parse_output(self, output_str: str) -> Grid:
        if not output_str.endswith(self.output_tail_token):
            raise ValueError(f"Output section must end with {self.output_tail_token}")
        output_str = output_str[: -len(self.output_tail_token)]
        return self.parse_grid(output_str)

    def _parse_examples(self, raw: str) -> list[Example]:
        """Parse a string containing multiple input/output example pairs.

        Args:
            raw: String containing concatenated input/output pairs

        Returns:
            List of Example objects containing the parsed grids
        """
        # Remove preprompt if present
        if raw.startswith(self.preprompt):
            raw = raw[len(self.preprompt) :]

        # Split on input token to separate examples
        example_strings = [ex for ex in raw.split(Token.INPUT) if ex]

        examples = []
        for ex_str in example_strings:
            # Re-add the input token that was removed by split
            full_example = Token.INPUT + ex_str
            input_grid, output_grid = self._parse_input_output(full_example)

            if output_grid is None:
                raise ValueError(f"Missing output grid in example: {full_example}")

            examples.append(Example(input_=input_grid, output=output_grid))
        return examples

    def format_task(self, task: Task, include_test_output: bool) -> str:
        train_text = "".join(
            [self._format_input_output(x.input_, x.output) for x in task.train_set]
        )
        test_example = task.test_set[0]
        test_text = self._format_input_output(
            test_example.input_, test_example.output if include_test_output else None
        )
        return f"{self.preprompt}{train_text}{test_text}"

    def parse_task(self, raw: str) -> Task:
        """Parse a complete task string including training and test examples.

        Args:
            raw: String containing preprompt, training examples, and test example

        Returns:
            Task object containing the parsed examples
        """
        # Get all examples
        all_examples = self._parse_examples(raw)

        if len(all_examples) < 2:  # Need at least 1 train and 1 test
            raise ValueError("Task must contain at least one training and test example")

        # Last example is test, rest are training
        test_example = all_examples[-1]
        train_examples = all_examples[:-1]

        return Task(
            id=None,  # ID not encoded in serialized format
            train_set=train_examples,
            test_set=[test_example],
        )

    # below are other formats that are targeteted used case
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

    def parse_test_output_grid(self, raw: str) -> Grid:
        examples = self._parse_examples(raw)
        return examples[-1].output
