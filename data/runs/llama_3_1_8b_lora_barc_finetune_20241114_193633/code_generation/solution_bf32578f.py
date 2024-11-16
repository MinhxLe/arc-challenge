from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color inversion, mirroring

# description:
# In the input, you will see a colored shape on a black background. To make the output, create a mirrored version of the shape across a central vertical axis, 
# and fill in the mirrored area with the same colors as the original shape.

def transform(input_grid):
    # Get the dimensions of the input grid
    height, width = input_grid.shape

    # Create an output grid filled with the background color
    output_grid = np.full((height, width), Color.BLACK)

    # Find the center line for mirroring
    center_x = width // 2

    # Iterate through the input grid to find non-background pixels
    for x in range(height):
        for y in range(width):
            if input_grid[x, y]!= Color.BLACK:
                # Mirror the color to the corresponding position in the output grid
                mirrored_y = (center_x - (y - center_x)) + center_x
                if 0 <= mirrored_y < width:  # Ensure we are within bounds
                    output_grid[x, mirrored_y] = input_grid[x, y]

    return output_grid