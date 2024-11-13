from pydantic.main import BaseModel
import inspect
import arckit
from arc import core
from arc.tasks import lib
from arc.utils import print_color_array


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

programming_addendum = f""" Additionally you are an expert Python programmer. You adhere to best practices to make code clear and easy to understand. This includes comments, docstring, and using constants. Where you can, you will use existing code such as the provided library.

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


def create_puzzle_descriptions_prompt(concepts: list[core.Concept], count: int) -> str:
    concept_string = ", ".join([c.value for c in concepts])
    return f"""
You are given high level concepts that should guide your creation of a new puzzle where the transformation law from inputs to outputs captures the concepts.
Note that inputs themselves do not need to conform to the concepts. For example, for the concept "reflection", the inputs do not need to be symmetric themselves.
Instead, the concepts should be embodied in how the input is transformed into the output.
You will create a description for the new puzzle that describes the input grids and the transformation law. The description
of the input grids should define inputs such that when the transformation law is applied to them, the concepts are illustrated.


For example:
# Example 1:
## concepts: color mapping, pattern replication
## new puzzle description:
In the input you will see a 3x3 grid of colored pixels. The colors are either black or gray. The output should be
a grid where:
1. If the pixel is black, it remains black in the output.
2. If the pixel is grey, it should be replaced by a 2x2 block of blue and red pixels in a checkerboard pattern.


# Example 2:
## concepts: color transformation, grid sections, boundary detection
## new puzzle description:
In the input, you will see a grid with a pattern of yellow and blue pixels with a black background, 
and multiple colored circles (not yellow or blue) placed randomly within the grid.
The goal is to transform the input grid by replacing all the blue pixels with the color of the closest circle 
and keeping the yellow pixels unchanged.

Give {count} different output description(s) for the input concepts: {concept_string}.
"""


class PuzzleDescriptions(BaseModel):
    descriptions: list[str]


def create_puzzle_code_prompt(description: str) -> str:
    return f"""
You will be given a description of a puzzle. This describes the valid input grids and transformation law. You will implement 
the following python functions.
1. A function that returns a random valid input grid based on the description. `def generate_input() -> Grid`. 
2. A function that implements the described transformation `def solve(input: Grid) -> Grid`. It is important that the function be deterministic.
You will have access to the specified code. Generated inputs from `generate_input`, when `solve` is applied, should generate interesting output
that captures the description.
```

The description is {description}. Only return the python code.
"""


def create_base_task_puzzle_descriptions_prompt(
    concept: core.Concept, count: int
) -> str:
    concept_string = concept.value

    return f"""
As a way to introduce student puzzle solvers to basic concepts, you want to generate simple puzzles that illustrate a
given concept in as pure a manner as possible. You take a single high level concept as an input, and then create a new puzzle where
the transformation law captures the concept in as clear and simplified a manner as possible. Note that input grids themselves
do not need to conform to the concepts. For example, for the concept "reflection", the input grids do not need to be symmetric themselves.
Instead, the concepts should be embodied in how the input grid is transformed into the output grid.

Given the input concept, you will output a description for the simple puzzle. The output description should describe
input grids and the transformation law. The description of the input grids should define inputs such that when the
transformation law is applied to them, the concept is illustrated.
The output description should not start by restating the input concept.

For example:
# Example 1:
## input concept: reflection
## output description:
In the input, you will see a grid of colored pixels. The output should be
a grid that is the reflection of the input grid across its horizontal midline.


# Example 2:
## given concept: rotation
## output description:
In the input, you will see a grid of colored pixels. The output should be
a grid that is the input grid rotated 90 degrees counterclockwise.

# Example 3:
## given concept: filling
## output description:
In the input, you will see a single four pixel by four pixel box whose boundary is
yellow and whose interior is black. The output grid should be the same as the input
grid except that the interior of the box should be filled in with yellow.

Give {count} different puzzle description(s) for the input concept: {concept_string}.
"""


def create_input_string(example) -> str:
    return "Input:" + "\n" + print_color_array(example[0])


def create_input_output_string(example) -> str:
    return (
        create_input_string(example)
        + "\n\n"
        + "Output:"
        + "\n"
        + print_color_array(example[1])
    )


def create_training_prompt(task: arckit.Task) -> str:
    return (
        """Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as strings, with cells (colors) separated by spaces and rows by newlines.
Here are the input and output grids for the reference examples:"""
        + "\n"
        + "\n\n\n".join(
            [
                f"Example {idx+1}:" + "\n" + create_input_output_string(example)
                for idx, example in enumerate(task.train)
            ]
        )
        + "\n\n\n"
        + "Here is the input grid for the test example:"
        + "\n"
        + create_input_string(task.test[0])
        + "\n\n"
        + "Write a Python function `transform` that can convert any given input grid to its corresponding output grid based on the pattern observed in the reference examples."
    )
