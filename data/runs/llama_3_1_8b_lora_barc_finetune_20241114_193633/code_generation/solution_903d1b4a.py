from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# reflection, symmetry, color mapping

# description:
# In the input, you will see a grid filled with colored pixels and a black background.
# The task is to reflect the colors across the vertical center line of the grid. 
# If a color is located at (x, y), it should be mirrored to (width - 1 - x, y) in the output grid.
# The colors that are not in the left half of the grid should be copied to the right half.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Get the dimensions of the input grid
    height, width = input_grid.shape
    output_grid = np.copy(input_grid)

    # Iterate through each pixel in the left half of the grid
    for x in range(width // 2):
        for y in range(height):
            color = input_grid[y, x]
            # Calculate the mirrored position in the right half
            mirrored_x = width - 1 - x
            # Copy the color to the mirrored position if it's not the background
            if color!= Color.BLACK:
                output_grid[y, mirrored_x] = color

    return output_grid