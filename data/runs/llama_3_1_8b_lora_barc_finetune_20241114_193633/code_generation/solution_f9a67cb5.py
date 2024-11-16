from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel rotation, color mapping

# description:
# In the input, you will see a grid with colored pixels forming a shape that has rotational symmetry. 
# To make the output, rotate the shape by 90 degrees clockwise and fill the new positions with the same color as the original shape.

def transform(input_grid):
    # Get the dimensions of the input grid
    height, width = input_grid.shape

    # Create an output grid initialized with the background color
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Rotate the grid by 90 degrees clockwise
    for x in range(height):
        for y in range(width):
            if input_grid[x, y]!= Color.BLACK:  # Only rotate non-background pixels
                new_x = y
                new_y = height - 1 - x
                output_grid[new_x, new_y] = input_grid[x, y]

    return output_grid