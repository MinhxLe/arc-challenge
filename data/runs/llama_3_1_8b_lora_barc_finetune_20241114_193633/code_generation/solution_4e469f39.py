from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, color filling, mirroring

# description:
# In the input, you will see a grid with a colored shape on a black background.
# To make the output, check for any horizontal, vertical, or diagonal symmetry in the shape.
# If symmetry is detected, fill the entire shape with a contrasting color (for example, red).
# If no symmetry is detected, leave the shape unchanged.

def transform(input_grid):
    # Detect horizontal, vertical, and diagonal symmetries in the input grid
    horizontal_symmetries = detect_translational_symmetry(input_grid, ignore_colors=[Color.BLACK])
    vertical_symmetries = detect_translational_symmetry(np.transpose(input_grid), ignore_colors=[Color.BLACK])
    diagonal_symmetries = detect_translational_symmetry(np.fliplr(np.transpose(input_grid)), ignore_colors=[Color.BLACK])

    # Create a copy of the input grid for output
    output_grid = np.copy(input_grid)

    # Check for horizontal symmetry
    if horizontal_symmetries:
        # Fill the shape with red if horizontal symmetry is detected
        for x in range(input_grid.shape[0]):
            for y in range(input_grid.shape[1]):
                if input_grid[x, y]!= Color.BLACK:
                    output_grid[x, y] = Color.RED

    # Check for vertical symmetry
    if vertical_symmetries:
        # Fill the shape with red if vertical symmetry is detected
        for x in range(input_grid.shape[0]):
            for y in range(input_grid.shape[1]):
                if input_grid[x, y]!= Color.BLACK:
                    output_grid[x, y] = Color.RED

    # Check for diagonal symmetry
    if diagonal_symmetries:
        # Fill the shape with red if diagonal symmetry is detected
        for x in range(input_grid.shape[0]):
            for y in range(input_grid.shape[1]):
                if input_grid[x, y]!= Color.BLACK:
                    output_grid[x, y] = Color.RED

    return output_grid