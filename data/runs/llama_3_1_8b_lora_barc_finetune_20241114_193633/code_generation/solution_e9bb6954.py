from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel counting, pattern transformation

# description:
# In the input, you will see a grid with several colored pixels. 
# To make the output, count the number of colored pixels in each column of the grid 
# and create a vertical bar chart on the right side of the grid, representing the counts 
# as vertical bars of the same colors as the original pixels.

def transform(input_grid):
    # Get the dimensions of the input grid
    n, m = input_grid.shape

    # Create an output grid with the same shape as input and filled with the background color
    output_grid = np.full((n, m), Color.BLACK)

    # Iterate through each column in the input grid
    for col in range(m):
        # Get the colors in the current column
        column_colors = input_grid[:, col]
        
        # Count the number of non-background pixels
        color_count = np.sum(column_colors!= Color.BLACK)

        # Fill the output grid with the counted colors in the same column
        for i in range(color_count):
            output_grid[i, col] = column_colors[i]

    return output_grid