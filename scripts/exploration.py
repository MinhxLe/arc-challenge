from typing import Tuple, List
import arckit
from datasets import load_dataset
from arc.code_gen import Program
from dataclasses import dataclass
from arc.core import Grid
from arc.external import openai
from arc.tasks import lib
import inspect


@dataclass
class GeneratedTask:
    task: arckit.Task
    program: Program


def parse_generated_task(messages) -> GeneratedTask | None:
    return None


prompt = f"""
You are a python programmer expert and teacher trying to teach students on how to fix bugs in programs. The types of program you are working with are transform functions that define a deterministc rule for a puzzle. Each puzzle consists of uncovering a deterministic rule, pattern, procedure,
algorithm, or transformation law that maps inputs to outputs.
Both the inputs and outputs are 2D grids of colored pixels. There are 10
colors, but the order of the colors is never relevant to the puzzle. Valid colors include 
black, blue, red, yellow, purple, orange, green, brown, grey, and pink. The grid height and
width can be between 1 to 30 pixels inclusive.

You will be given a program that has a description of the intended transformation as well as an implementation of the transform function. Additionally you have access to a arc.dsl.common library which the student has access to as well.

Your goal is to modify the program to introduce a bug or regression into the code. You should not indicate where the changes is so that the student is able to learn how to fix the issue themselves. Such regressions, for example, can be using the wrong constant, invoking the wrong helper function, removal of key lines of code. The function should not crash when invoked - instead the output should change. You should describe the introduced regression and how the specification is no longer met as a result.

The library `arc/dsl/common.py` is provided
```python
{inspect.getsource(lib)}
```

Here is the original puzzle and solution implementation.
```python
{source}
```

Output the modified program with an introduced regression.
"""

response = openai.complete(prompt, temperature=1)
