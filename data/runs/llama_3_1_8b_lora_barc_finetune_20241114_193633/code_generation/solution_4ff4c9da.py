from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, shape transformation

# description:
# In the input, you will see a grid filled with a mixture of colored pixels and a single black pixel acting as a reference point.
# To create the output, apply a transformation that changes each color in the grid to a new color based on its distance from the reference point.
# Each color in the grid will be transformed to a new color based on the following mapping:
# - If the distance to the reference point is even, change to Color.BLUE.
# - If the distance to the reference point is odd, change to Color.RED.
# The output grid will maintain the same shape as the input grid.

def transform(input_grid):
    # Find the position of the black reference pixel
    reference_pixel_location = np.argwhere(input_grid == Color.BLACK)
    assert len(reference_pixel_location) == 1
    reference_x, reference_y = reference_pixel_location[0]

    # Create an output grid initialized to the same shape as input_grid
    output_grid = np.copy(input_grid)

    # Apply the transformation based on the distance to the reference point
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            if input_grid[x, y]!= Color.BLACK:  # Skip the reference pixel
                distance = abs(x - reference_x) + abs(y - reference_y)
                if distance % 2 == 0:
                    output_grid[x, y] = Color.BLUE
                else:
                    output_grid[x, y] = Color.RED

    return output_grid