from functools import cached_property
import re
from rich.console import Console
from rich.table import Table
import arckit
from dataclasses import dataclass
from typing import Callable
from arc import core
import numpy as np


@dataclass
class Program:
    source: str
    fn: Callable[[core.Grid], core.Grid] | None
    interpret_error: Exception | None

    @classmethod
    def from_source(cls, source: str) -> "Program":
        fn = None
        interpret_error = None
        try:
            local = dict()
            exec(source, local)
            assert "transform" in local
            fn = local["transform"]
        except Exception as e:
            interpret_error = e

        return Program(source, fn, interpret_error)

    def call(self, input_: core.Grid) -> core.Grid | Exception:
        # to prevent side effects
        input_ = np.copy(input_)
        if self.fn is None:
            assert self.interpret_error is not None
            return self.interpret_error
        try:
            output = self.fn(input_)
            # this is kind of jank but we want to do sanitization
            assert np.all(output < len(core.Color))
            assert np.all(output >= 0)
            return output
        except Exception as e:
            return e


@dataclass
class ProgramExecution:
    program: Program
    task: arckit.Task

    @cached_property
    def executions(self) -> list[core.Grid | Exception]:
        evaluations = [self.program.call(input_) for input_, _ in self.task.train]
        for input_, _ in self.task.train:
            evaluations.append(self.program.call(input_))
        return evaluations

    @cached_property
    def training_success(self) -> bool:
        training_successes = []
        for (_, output_), evaluation in zip(self.task.train, self.executions):
            if isinstance(evaluation, Exception):
                this_training_success = False
            else:
                this_training_success = np.array_equal(output_, evaluation)
            training_successes.append(this_training_success)
        return all(training_successes)

    def create_result_table(self) -> Table:
        table = Table()
        actual = self.executions
        expected = [t[1] for t in self.task.train]
        table.add_row(*[arckit.fmt_grid(x) for x in actual])
        table.add_section()
        table.add_row(*[arckit.fmt_grid(x) for x in expected])
        return table

    def show(self):
        console = Console()
        console.print(self.create_result_table())


def remove_comments(source_code: str) -> str:
    """
    Removes all comments from Python source code while preserving functionality.
    Handles both inline (#) and multi-line (''') comments.

    Args:
        source_code (str): Valid Python source code as a string

    Returns:
        str: Source code with all comments removed
    """

    # First, handle multi-line strings/comments
    def replace_multiline_strings(match):
        # Preserve actual multi-line strings (not comments) by counting quotes before
        line_start = source_code.rfind("\n", 0, match.start()) + 1
        if line_start == 0:
            line_start = 0
        line_before = source_code[line_start : match.start()].strip()

        # If it's part of an assignment or expression, keep it
        if line_before and (
            line_before[-1] in "=[({,"
            or any(line_before.startswith(x) for x in ["return", "yield"])
        ):
            return match.group(0)
        # Otherwise, it's a comment - replace with newlines to preserve line numbers
        return "\n" * match.group(0).count("\n")

    # Handle multi-line strings/comments
    pattern = r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\''
    code = re.sub(pattern, replace_multiline_strings, source_code)

    # Handle single-line comments while preserving strings
    result = []
    lines = code.split("\n")
    in_string = False
    string_char = None

    for line in lines:
        if not line.strip():
            result.append(line)
            continue

        new_line = []
        i = 0
        while i < len(line):
            char = line[i]

            # Handle string literals
            if char in "\"'":
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char and line[i - 1] != "\\":
                    in_string = False
                new_line.append(char)

            # Handle comments
            elif char == "#" and not in_string:
                break

            else:
                new_line.append(char)

            i += 1

        result.append("".join(new_line).rstrip())

    return "\n".join(result)
