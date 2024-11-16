from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, diagonal reflection

# description:
# In the input grid, you will see a pattern of colored pixels on a black background. 
# To create the output, reflect the colored pixels diagonally across the grid, creating a mirrored pattern of the same color.
# The reflection should be done across both the main diagonal (top-left to bottom-right) and the anti-diagonal (top-right to bottom-left).
# Any pixels that overlap with the original colors should retain their colors.

def transform(input_grid):
    # Create an output grid initialized to black
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Iterate through the grid to find colored pixels
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            if input_grid[x, y]!= Color.BLACK:
                # Get the color of the current pixel
                color = input_grid[x, y]

                # Calculate mirrored positions
                output_grid[x, y] = color  # Original position
                output_grid[y, x] = color  # Diagonal reflection across main diagonal
                output_grid[x, 2 * x - y] = color  # Diagonal reflection across anti-diagonal

    return output_grid