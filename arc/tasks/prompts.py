from pydantic.main import BaseModel


system_prompt = """
You are a puzzle maker designing geometric, physical, and topological
puzzles for curious middle-schoolers.

Each puzzle consists of uncovering a deterministic rule, pattern, procedure,
algorithm, or transformation law that maps inputs to outputs.
Both the inputs and outputs are 2D grids of colored pixels. There are 10
colors, but the order of the colors is never relevant to the puzzle. Valid colors include 
black, blue, red, yellow, purple,  orange, green, brown, grey, and pink. The grid height and
width can be inclusive between 1 to 30 pixels.
"""


def create_puzzle_descriptions_prompt(concepts: list[str], count: int) -> str:
    concept_string = ", ".join(concepts)
    return f"""
You are given high level concepts in order to create a new puzzle capturing those concepts. You will create
a description for the new puzzle that describes the valid input grids and the transformation law.

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
