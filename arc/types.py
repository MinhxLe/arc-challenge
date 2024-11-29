from functools import cached_property
from rich.console import Console
from rich.table import Table
import arckit
from dataclasses import dataclass
from typing import Callable
from arc import core
import numpy as np


@dataclass
class Evaluation:
    input_: core.Grid
    output: core.Grid | Exception


@dataclass
class Program:
    task: arckit.Task
    source: str
    fn: Callable[[core.Grid], core.Grid]

    @cached_property
    def evaluations(self) -> list[core.Grid | Exception]:
        evaluations = []
        for input_, _ in self.task.train:
            try:
                output = self.fn(input_)
            except Exception as e:
                output = e
            evaluations.append(output)
        return evaluations

    @cached_property
    def training_success(self) -> bool:
        training_successes = []
        for (_, output_), evaluation in zip(self.task.train, self.evaluations):
            if isinstance(evaluation, Exception):
                this_training_success = False
            else:
                this_training_success = np.array_equal(output_, evaluation)
            training_successes.append(this_training_success)
        return all(training_successes)

    def create_result_table(self) -> Table:
        table = Table()
        actual = self.evaluations
        expected = [t[1] for t in self.task.train]
        table.add_row(*[arckit.fmt_grid(x) for x in actual])
        table.add_section()
        table.add_row(*[arckit.fmt_grid(x) for x in expected])
        return table

    def show(self):
        console = Console()
        console.print(self.create_result_table())
