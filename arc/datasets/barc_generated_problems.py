"""
utils in parsing/transforming BARC dataset
"""

import arckit
from dataclasses import dataclass
import numpy as np
import re
from typing import Callable, List, Tuple
from datasets import load_dataset
from arc.core import Grid, Color
from collections import defaultdict

_DATASET_NAME = "barc0/induction_100k-gpt4-description-gpt4omini-code_generated_problems_messages_format_0.3"


def get_dataset():
    return load_dataset(_DATASET_NAME)


@dataclass
class Program:
    source: str
    fn: Callable[[Grid], Grid]


@dataclass
class GeneratedTask:
    description: str
    task: arckit.Task
    program: Program


def extract_generated_task(msgs: list[dict]) -> GeneratedTask:
    assert len(msgs) == 3
    description = _extract_description(msgs[2]["content"])
    program = _extract_program(msgs[2]["content"])
    train_input, _, test_input = _extract_grids(msgs[1]["content"])
    # we don't used the parsed trained output
    task = arckit.Task(
        id=None,
        train=[dict(input=i, output=program.fn(i)) for i in train_input],
        test=[dict(input=test_input, output=program.fn(test_input))],
    )

    return GeneratedTask(
        description=description,
        task=task,
        program=program,
    )


def _extract_description(msg: str) -> str:
    """
    Extracts the description from Python code comments that appear between
    '# description:' and the next non-comment line or different comment section.

    Args:
        code (str): Python code containing commented description

    Returns:
        str: The extracted description with comment markers removed and proper spacing
    """
    # Split the code into lines
    lines = msg.split("\n")

    description = []
    in_description = False

    for line in lines:
        stripped = line.strip()

        # Start collecting when we hit the description marker
        if stripped.startswith("# description:"):
            in_description = True
            # Add the text after "# description:" if there is any
            desc_text = stripped[len("# description:") :].strip()
            if desc_text:
                description.append(desc_text)
            continue

        # If we're in description section and hit a comment, add it
        if in_description and stripped.startswith("#"):
            # Remove the comment marker and any leading/trailing whitespace
            desc_text = stripped[1:].strip()
            if desc_text:
                description.append(desc_text)
        # Stop when we hit a non-comment line or different comment section
        elif in_description:
            break

    # Join the lines with proper spacing
    return " ".join(description)


def _extract_program(msg: str) -> Program:
    # Extract code between Python code blocks
    code_match = re.search(r"```python\n(.*?)```", msg, re.DOTALL)
    assert code_match is not None
    source_code = (
        code_match.group(1)
        .strip()
        .replace("from common import", "from arc.dsl.common import")
    )

    local = dict()
    exec(source_code, local)
    assert "transform" in local
    fn = local["transform"]
    return Program(source_code, fn)


def _extract_grids(
    text,
) -> Tuple[List[Grid], List[Grid], Grid]:
    """
    extract input and output grids from the example text and return them as lists of 2D arrays,
    along with the test input grid.

    Args:
        text (str): The input text containing the grid examples

    Returns:
        tuple: Three items:
            - List of example input grids (2D arrays of color strings)
            - List of example output grids (2D arrays of color strings)
            - Test input grid (2D array of color strings)
    """
    # Lists to store the extractd grids
    example_inputs = []
    example_outputs = []
    test_input = None

    # Split text into examples (split by "Example" or "Input:")
    parts = text.split("Example")[1:]  # Skip the initial text before first example

    # Process the example pairs
    for example in parts:
        # Skip if this part doesn't contain "Input:"
        if "Input:" not in example:
            continue

        # Find the input and output sections
        input_start = example.find("Input:") + 6
        output_start = example.find("Output:") + 7

        # Extract and extract input grid
        input_end = example.find("\n\n", input_start)
        if input_end == -1 or (output_start - 7 > 0 and input_end > output_start):
            input_end = output_start - 7 if output_start - 7 > 0 else len(example)
        input_text = example[input_start:input_end].strip()
        input_grid = [row.split() for row in input_text.split("\n")]

        # If there's no Output section, this is the test case
        if output_start - 7 == -1:
            test_input = input_grid
            continue

        # Add to example inputs
        example_inputs.append(input_grid)

        # extract output grid
        next_section = example.find("\n\n", output_start)
        if next_section == -1:
            next_section = len(example)
        output_text = example[output_start:next_section].strip()
        output_grid = [row.split() for row in output_text.split("\n")]
        example_outputs.append(output_grid)

    # Find test input if it wasn't in the examples
    test_part = text.split("Here is the input grid for the test example:")[1]
    test_start = test_part.find("Input:") + 6
    test_end = test_part.find("\n\n", test_start)
    if test_end == -1:
        test_end = len(test_part)
    test_text = test_part[test_start:test_end].strip()
    test_input = [row.split() for row in test_text.split("\n")]

    return (
        [_color_grid_to_int_grid(x) for x in example_inputs],
        [_color_grid_to_int_grid(x) for x in example_outputs],
        _color_grid_to_int_grid(test_input),
    )


def _color_grid_to_int_grid(raw_grid: List[List[str]]) -> Grid:
    """
    Convert a 2D list of color strings to a numpy array of integer values based on Color enum.
    Handles case-insensitive color names and provides helpful error messages for invalid colors.

    Args:
        raw_grid: List[List[str]] - 2D list of color strings

    Returns:
        np.ndarray - 2D numpy array of integer values representing colors

    Raises:
        ValueError: If an invalid color name is encountered
    """
    # Create a case-insensitive mapping of color names to enum values
    color_map = dict(
        BLACK=Color.BLACK,
        BLUE=Color.BLUE,
        RED=Color.RED,
        GREEN=Color.GREEN,
        YELLOW=Color.YELLOW,
        GREY=Color.GREY,
        GRAY=Color.GRAY,
        PINK=Color.PINK,
        ORANGE=Color.ORANGE,
        TEAL=Color.TEAL,
        MAROON=Color.MAROON,
    )
    # Get grid dimensions
    if not raw_grid or not raw_grid[0]:
        return np.array([], dtype=np.int32)

    height = len(raw_grid)
    width = len(raw_grid[0])

    # Initialize output array
    int_grid = np.zeros((height, width), dtype=np.integer)

    # Convert each color string to its integer value
    for i in range(height):
        for j in range(width):
            color_str = raw_grid[i][j].upper()
            if color_str in color_map:
                color_val = color_map[color_str]
            else:
                raise ValueError(
                    f"Invalid color '{raw_grid[i][j]}' at position ({i}, {j}). "
                )

            int_grid[i, j] = color_val
    return int_grid
