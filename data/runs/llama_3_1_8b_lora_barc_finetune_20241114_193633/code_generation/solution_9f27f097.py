from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, reflection, mirroring

# description:
# In the input, you will see a grid with a colorful shape and a black background.
# To make the output, identify the axis of symmetry (horizontal, vertical, or both)
# and reflect the shape across that axis, filling in the mirrored shape in the output grid.

def transform(input_grid):
    # Create an output grid initialized to the background color (black)
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Find the axis of symmetry
    horizontal_symmetry = detect_translational_symmetry(input_grid, ignore_colors=[Color.BLACK])
    vertical_symmetry = detect_translational_symmetry(input_grid, ignore_colors=[Color.BLACK])

    # Check for horizontal symmetry
    if horizontal_symmetry is not None:
        # Reflect the shape vertically
        for x, y in np.argwhere(input_grid!= Color.BLACK):
            mirrored_x = x
            mirrored_y = (2 * input_grid.shape[1] - 1) - y
            if mirrored_x >= 0 and mirrored_y >= 0:
                output_grid[mirrored_x, mirrored_y] = input_grid[x, y]

    # Check for vertical symmetry
    if vertical_symmetry is not None:
        # Reflect the shape horizontally
        for x, y in np.argwhere(input_grid!= Color.BLACK):
            mirrored_x = (2 * input_grid.shape[0] - 1) - x
            mirrored_y = y
            if mirrored_x >= 0 and mirrored_y >= 0:
                output_grid[mirrored_x, mirrored_y] = input_grid[x, y]

    # Check for both symmetries
    if horizontal_symmetry is not None and vertical_symmetry is not None:
        # Reflect the shape diagonally
        for x, y in np.argwhere(input_grid!= Color.BLACK):
            mirrored_x = (2 * input_grid.shape[0] - 1 - x)
            mirrored_y = (2 * input_grid.shape[1] - 1 - y)
            if mirrored_x >= 0 and mirrored_y >= 0:
                output_grid[mirrored_x, mirrored_y] = input_grid[x, y]

    # Copy original shape to output grid
    output_grid[input_grid!= Color.BLACK] = input_grid[input_grid!= Color.BLACK]

    return output_grid