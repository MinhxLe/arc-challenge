from pydantic.main import BaseModel
import inspect
from arc import core
from arc.tasks import lib

puzzlemaker_role_prompt = """
You are a puzzle maker designing geometric, physical, and topological
visual puzzles for curious middle-schoolers.

Each puzzle consists of uncovering a deterministic rule, pattern, procedure,
algorithm, or transformation law that maps inputs to outputs.
Both the inputs and outputs are 2D grids of colored pixels. There are 10
colors, but the order of the colors is never relevant to the puzzle. Valid colors include 
black, blue, red, yellow, purple,  orange, green, brown, grey, and pink. The grid height and
width can be inclusive between 1 to 30 pixels.
"""

programmer_role_prompt = f"""
{puzzlemaker_role_prompt}. Additionally you are an expert python programmer. You adhere to best practices to make code clear and easy to understand. This includes comments, docstring, using and using constants. Where you can, you will use existing code such as the provided library.

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


def create_puzzle_descriptions_prompt(concepts: list[core.Concept], count: int) -> str:
    concept_string = ", ".join([c.value for c in concepts])
    return f"""
You are given high level concepts in order to create a new puzzle where the transformation law captures the concepts. Note that the concepts
do NOT apply to the inputs. You will create a description for the new puzzle that describes the input grids and the transformation law. The description
of input grids be inputs which the outputs that exemplify transformation law.


For example:
# Example 1:
## input concept: color mapping, pattern replication
## output description:
In the input you will see a 3x3 grid of colored pixels. The colors are either black or gray. The output should be
a grid where:
1. If the pixel is black, it remains black in the output.
2. If the pixel is gray, it shoudl be replaced by a 2x2 block of blue and red pixel in a checkerboard pattern.


# Example 2:
## input concept: color transformation, grid sections, boundary detection
## output description:
In the input, you will see a grid with a pattern of yellow and blue pixels with a black background, 
and multiple colored circles (not yellow or blue) placed randomly within the grid.
The goal is to transform the output grid by replacing all the blue pixels with the color of the closest circle 
and keeping the yellow pixels unchanged.

Give {count} different output description(s) for the input concepts: {concept_string}.
"""


class PuzzleDescriptions(BaseModel):
    descriptions: list[str]


def create_puzzle_code_prompt(description: str) -> str:
    return f"""
You will be given a description of a puzzle. This describes the valid input grids and transformation law. You will implement
a the following python functions.
1. A function that returns a random valid input grid based on the description. `def generate_input() -> Grid`. 
2. A function that implements the described transformation `def solve(input: Grid) -> Grid`. It is important the function is deterministic.
You will have access to the specified code. Generated inputs from `generate_input`, when `solve` is applied, should generating interesting output
that captures the description.
```

The description is {description}. Only return the python code.
"""
