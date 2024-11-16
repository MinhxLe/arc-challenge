from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# rotation, symmetry detection, color mapping

# description:
# In the input, you will see a grid with a pattern of colored pixels. 
# To create the output, you should rotate the entire grid 90 degrees clockwise and check if the new grid exhibits rotational symmetry.
# If it does, color the entire grid with the color of the original center pixel. 
# If it does not, color the entire grid black.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Rotate the grid 90 degrees clockwise
    rotated_grid = np.rot90(input_grid, k=-1)

    # Check for rotational symmetry in the rotated grid
    sym = detect_rotational_symmetry(rotated_grid, ignore_colors=[Color.BLACK])
    if sym:
        center_x, center_y = sym.center_x, sym.center_y
        center_color = rotated_grid[center_x, center_y]
        output_grid = np.full(rotated_grid.shape, center_color)
    else:
        output_grid = np.full(rotated_grid.shape, Color.BLACK)

    return output_grid