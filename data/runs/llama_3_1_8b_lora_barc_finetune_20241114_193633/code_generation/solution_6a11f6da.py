from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color extraction, quadrant separation

# description:
# In the input, you will see a grid filled with colored pixels, where the pixels are arranged in four quadrants.
# To create the output, extract the colors from each quadrant and fill a new grid with these colors in a specified order: 
# top-left quadrant at the top, top-right quadrant in the middle row, bottom-left quadrant in the middle row,
# and bottom-right quadrant below the middle row.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Get the size of the input grid
    n, m = input_grid.shape

    # Determine the midpoints
    mid_x, mid_y = n // 2, m // 2

    # Create an output grid with the same shape as the input grid
    output_grid = np.full((n, m), Color.BLACK)

    # Extract colors from each quadrant
    top_left = input_grid[:mid_x, :mid_y]
    top_right = input_grid[:mid_x, mid_y:]
    bottom_left = input_grid[mid_x:, :mid_y]
    bottom_right = input_grid[mid_x:, mid_y:]

    # Fill the output grid with extracted colors
    output_grid[:mid_x, :mid_y] = top_left[top_left!= Color.BLACK]  # Top-left
    output_grid[:mid_x, mid_y:] = top_right[top_right!= Color.BLACK]  # Top-right
    output_grid[mid_x:, :mid_y] = bottom_left[bottom_left!= Color.BLACK]  # Bottom-left
    output_grid[mid_x:, mid_y:] = bottom_right[bottom_right!= Color.BLACK]  # Bottom-right

    return output_grid