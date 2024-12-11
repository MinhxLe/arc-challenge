import inspect
import arckit
from arc import core
from arc.tasks import lib
from arc.utils import create_color_array
from arc.program import ProgramExecution
import numpy as np


puzzlemaker_role_prompt = """
You are a puzzle maker designing geometric, physical, and topological
visual puzzles for curious middle-schoolers.

Each puzzle consists of uncovering a deterministic rule, pattern, procedure,
algorithm, or transformation law that maps inputs to outputs.
Both the inputs and outputs are 2D grids of colored pixels. There are 10
colors, but the order of the colors is never relevant to the puzzle. Valid colors include 
black, blue, red, yellow, purple, orange, green, brown, grey, and pink. The grid height and
width can be between 1 to 30 pixels inclusive.
"""

programming_addendum = f""" Additionally you are an expert Python programmer.
You adhere to best practices to make code clear and easy to understand.
This includes comments, docstring, and using constants.
Where you can, you will use existing code such as the two libraries provided below.

You are given access to the following python code.
arc/core.py
```python
{inspect.getsource(core)}
```

arc/tasks/lib.py
The following are just function signatures.
```python
{inspect.getsource(lib)}
```
"""

programmer_role_prompt = f"""
{puzzlemaker_role_prompt}. {programming_addendum}
"""

puzzlesolver_role_prompt = f"""You are a world-class puzzle solver with exceptional pattern recognition skills and expertise in Python programming.
Your task is to analyze puzzles and provide Python solutions. {programming_addendum}"""

addendum_for_nonfinetuned = """The function should accept a single argument of type Grid from arc.core.
Each element of Grid is an integer corresponding to the Color IntEnum from arc.core.
For ease of language processing, the input and output examples given above are written
using the color names, but the `transform` function should operate on type Grid.
It is okay for your response to include an explanation of the problem and
your approach, but your response must end with a code block that begins with:
```python"""


def create_solve_task_prompt(task: arckit.Task) -> str:
    training_examples = "\n\n\n".join(
        [
            f"Example {idx+1}:" + "\n" + _create_input_output_string(input_, output)
            for idx, (input_, output) in enumerate(task.train)
        ]
    )

    return f"""Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as strings, with cells (colors) separated by spaces and rows by newlines.
Here are the input and output grids for the reference examples:
{training_examples}


Here is the input grid for the test example:
{_create_input_string(task.test[0][0])}

Write a Python function `transform` that can convert any given input grid to its corresponding output grid based on the pattern observed in the reference examples.
"""


def _create_execution_string(execution: ProgramExecution) -> str:
    execution_results = []
    for i, ((input_, expected), output) in enumerate(
        zip(execution.task.train, execution.executions)
    ):
        if isinstance(output, Exception):
            execution_str = _create_input_error_string(input_, output)
            correct = False
        else:
            assert isinstance(output, np.ndarray)
            execution_str = _create_input_output_string(input_, output)
            correct = expected.shape == output.shape and np.all(expected == output)
        execution_results.append(
            f"{'Correct' if correct else 'Wrong'} Execution of Example {i+1}\n{execution_str}"
        )
    execution_result_str = "\n\n".join(execution_results)

    return f"""```python
{execution.program.source}
```

Here are execution results of this program.
{execution_result_str}"""


def create_improve_solve_task_prompt(
    task: arckit.Task,
    executions: list[ProgramExecution],
) -> str:
    training_examples = "\n\n\n".join(
        [
            f"Example {idx+1}:" + "\n" + _create_input_output_string(input_, output)
            for idx, (input_, output) in enumerate(task.train)
        ]
    )

    program_executions = "\n\n\n".join(
        [
            f"Attempt {i+1}\n{_create_execution_string(program)}"
            for i, program in enumerate(executions)
        ]
    )
    return f"""Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as strings, with cells (colors) separated by spaces and rows by newlines.
Here are the input and output grids for the reference examples:
{training_examples}


Here is the input grid for the test example:
{_create_input_string(task.test[0][0])}

You are to write a Python function `transform` that can convert any given input grid to its corresponding output grid based on the pattern observed in the reference examples.

Here are previous attempts of an implementation and their execution result.
{program_executions}

Iterate on previous versions of Python function `transform` so that there will be more correct executions.
Pay attention to the wrong executions of Example cases to discover and correct flaws in previous versions of `transform`. Output only the python code.
"""


def _create_input_string(input_: core.Grid) -> str:
    return f"Input: \n{create_color_array(input_)}"


def _create_input_output_string(input_: core.Grid, output: core.Grid) -> str:
    return f"""{_create_input_string(input_)}

Output:
{create_color_array(output)}"""


def _create_input_error_string(input_: core.Grid, error: Exception) -> str:
    return f"""{_create_input_string(input_)}

Error:
{str(error)}"""
