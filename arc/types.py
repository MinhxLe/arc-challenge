from functools import cached_property
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
            return self.fn(input_)
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
