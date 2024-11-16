from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color stacking, gravity, filling

# description:
# In the input, you will see a grid with various colored pixels on a black background. 
# To create the output, simulate gravity by shifting all colored pixels down to the lowest available position in their respective columns, 
# filling any empty spaces above them with the background color (black).

def transform(input_grid):
    # Create an output grid initialized to the background color (black)
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Get the dimensions of the grid
    n, m = input_grid.shape

    # Iterate through each column in the grid
    for col in range(m):
        # Create a list to hold the colors that need to be moved down
        column_colors = []

        # Collect colors in the current column
        for row in range(n):
            if input_grid[row, col]!= Color.BLACK:
                column_colors.append(input_grid[row, col])

        # Place the collected colors at the bottom of the column in the output grid
        for i, color in enumerate(column_colors):
            output_grid[n - 1 - i, col] = color

    return output_grid