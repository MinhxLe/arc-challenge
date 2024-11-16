from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# reflection, color mapping

# description:
# In the input, you will see a grid with a series of colored pixels arranged in a vertical line down the left side of the grid, 
# and a horizontal line of pixels of varying colors along the top row. 
# To create the output, reflect the colors of the vertical line horizontally across the horizontal line.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    # Get the colors from the vertical line on the left
    vertical_line_colors = output_grid[:, 0]
    # Get the colors from the horizontal line at the top
    horizontal_line_colors = output_grid[0, :]

    # Reflect the vertical line across the horizontal line
    for x in range(output_grid.shape[0]):
        for y in range(output_grid.shape[1]):
            if vertical_line_colors[x]!= Color.BLACK:  # Ignore the background
                # Find the corresponding position to reflect
                reflected_y = output_grid.shape[1] - 1 - y
                # Set the reflected pixel to the corresponding color
                output_grid[x, reflected_y] = horizontal_line_colors[x]

    return output_grid