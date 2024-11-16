from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry, mirroring, color filling

# description:
# In the input, you will see a series of colored pixels arranged in a row along one edge of the grid. 
# To create the output, mirror these colored pixels across the center of the grid, creating a symmetric pattern. 
# The output grid will be filled with the mirrored colors, leaving the rest of the grid as the background color.

def transform(input_grid):
    # Determine the dimensions of the input grid
    height, width = input_grid.shape
    
    # Create an output grid initialized to the background color
    output_grid = np.full((height, width), Color.BLACK)
    
    # Extract the colors from the top row of the input grid
    top_row_colors = input_grid[0, :]
    middle_x = height // 2
    middle_y = width // 2
    
    # Fill the output grid by mirroring the top row colors
    for x in range(height):
        for y in range(width):
            if input_grid[x, y]!= Color.BLACK:
                # Calculate the mirrored position
                mirrored_x = height - 1 - x
                mirrored_y = width - 1 - y
                # Fill the mirrored positions
                output_grid[x, y] = input_grid[x, y]  # Original position
                output_grid[mirrored_x, mirrored_y] = input_grid[x, y]  # Vertical mirror
                output_grid[mirrored_x, y] = input_grid[x, y]  # Horizontal mirror
                output_grid[x, mirrored_y] = input_grid[x, y]  # Diagonal mirror

    return output_grid