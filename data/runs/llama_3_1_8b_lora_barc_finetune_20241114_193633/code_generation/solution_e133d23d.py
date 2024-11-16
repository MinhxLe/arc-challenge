from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern detection, scaling, color mapping

# description:
# In the input, you will see a grid with a vertical band of colored pixels on the left edge and a vertical band of colored pixels on the right edge.
# To make the output, you should scale the left band of colors to fill the entire grid horizontally, and the right band of colors to fill the entire grid vertically.
# The output grid should have the same height as the input grid, but its width should be the same as the height of the left band.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Determine the number of rows and columns in the input grid
    rows, cols = input_grid.shape
    
    # Get the colors from the left and right edges
    left_colors = input_grid[:, 0]
    right_colors = input_grid[:, -1]
    
    # Create output grid
    output_grid = np.full((rows, rows), Color.BLACK)

    # Fill the left band horizontally
    for x in range(rows):
        output_grid[x, :] = left_colors[x]

    # Fill the right band vertically
    for y in range(rows):
        output_grid[y, :] = right_colors[y]

    return output_grid