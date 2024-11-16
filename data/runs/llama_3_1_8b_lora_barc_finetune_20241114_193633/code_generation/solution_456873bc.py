from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern extraction, symmetry, mirroring

# description:
# In the input, you will see a grid with a pattern of colored pixels. The pattern is symmetric about a vertical line. 
# To make the output, you should extract the left half of the pattern, mirror it to the right side of the grid, 
# and fill in the mirrored section with a specific color (green).

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid to avoid modifying the original
    output_grid = np.copy(input_grid)

    # Determine the width of the grid
    height, width = output_grid.shape

    # Determine the midpoint for mirroring
    mid_x = width // 2

    # Extract the left half of the grid
    left_half = output_grid[:, :mid_x]

    # Mirror the left half to the right side of the grid
    output_grid[:, mid_x:] = np.copy(left_half[:, ::-1])  # Mirror horizontally

    # Fill the mirrored section with a specific color (green)
    output_grid[:, mid_x:][:, :] = Color.GREEN

    return output_grid