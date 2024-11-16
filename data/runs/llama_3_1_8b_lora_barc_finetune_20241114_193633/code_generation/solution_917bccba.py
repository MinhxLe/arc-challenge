from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, color filling, bounding box

# description:
# In the input, you will see a grid with a pattern that has rotational symmetry around a central point.
# To create the output, fill in the missing parts of the pattern to complete the rotational symmetry, 
# using the color of the existing pattern.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid for output
    output_grid = np.copy(input_grid)

    # Detect the center of rotation
    sym = detect_rotational_symmetry(input_grid, ignore_colors=[Color.BLACK])

    # If no symmetry found, return the original grid
    if sym is None:
        return output_grid

    # Find all colored pixels
    colored_pixels = np.argwhere(input_grid!= Color.BLACK)

    # For each colored pixel, find its rotational positions and fill in the missing parts
    for x, y in colored_pixels:
        # Get the color of the current pixel
        color = input_grid[x, y]

        # Get the rotational symmetries
        rotations = sym.apply(x, y, iters=1)

        # Fill in the missing colors
        for rotated_x, rotated_y in rotations:
            if output_grid[rotated_x, rotated_y] == Color.BLACK:
                output_grid[rotated_x, rotated_y] = color

    return output_grid