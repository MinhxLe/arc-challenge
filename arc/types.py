from functools import cached_property
from rich.console import Console
from rich.table import Table
import arckit
from dataclasses import dataclass
from typing import Callable
from arc import core


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

    def show(self):
        console = Console()
        table = Table()
        actual = self.evaluations
        expected = [t[1] for t in self.task.train]
        table.add_row(*[arckit.fmt_grid(x) for x in actual])
        table.add_row(*[arckit.fmt_grid(x) for x in expected])
        console.print(table)
