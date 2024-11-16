from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, pixel manipulation

# description:
# In the input, you will see a grid with various colored objects and a gray border surrounding them.
# To create the output, you should color the gray border with the color of the nearest object in the same row or column.

def transform(input_grid):
    output_grid = np.copy(input_grid)
    rows, cols = input_grid.shape

    # Identify the colors in the grid excluding the gray border
    for r in range(rows):
        for c in range(cols):
            if input_grid[r, c]!= Color.GRAY and input_grid[r, c]!= Color.BLACK:
                color = input_grid[r, c]
                # Color the gray border in the same row
                output_grid[r, :] = np.where(output_grid[r, :] == Color.GRAY, color, output_grid[r, :])
                # Color the gray border in the same column
                output_grid[:, c] = np.where(output_grid[:, c] == Color.GRAY, color, output_grid[:, c])

    return output_grid