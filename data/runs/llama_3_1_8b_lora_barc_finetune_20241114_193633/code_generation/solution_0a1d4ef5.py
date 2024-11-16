from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, symmetry, reflection

# description:
# In the input, you will see a grid filled with a variety of colors, some of which are symmetrically placed.
# To create the output grid, you should reflect the colors across the vertical axis of the grid.
# The colors that are symmetrically placed should be reflected, and the output grid should retain the same structure as the input grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    # Get the dimensions of the input grid
    height, width = input_grid.shape

    # Loop through each pixel in the input grid
    for x in range(height):
        for y in range(width):
            # Check if the pixel is not black
            if input_grid[x, y]!= Color.BLACK:
                # Find the corresponding pixel on the opposite side of the grid
                reflected_x = x
                reflected_y = width - 1 - y

                # Set the reflected pixel to the same color
                output_grid[reflected_x, reflected_y] = input_grid[x, y]

    return output_grid