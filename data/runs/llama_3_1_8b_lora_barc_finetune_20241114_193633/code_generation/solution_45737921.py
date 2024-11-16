from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# rotation, color blending

# description:
# In the input, you will see a colored shape and a black background. 
# To make the output, rotate the shape 90 degrees clockwise and blend it with the original shape, 
# where overlapping pixels will be colored with the color of the shape that is on top.

def transform(input_grid):
    # Detect the original shape
    shape = input_grid
    shape_rotated = np.rot90(shape, k=-1)  # Rotate the shape 90 degrees clockwise

    # Create the output grid initialized to the original shape
    output_grid = shape.copy()

    # Blend the rotated shape onto the output grid
    for x in range(shape.shape[0]):
        for y in range(shape.shape[1]):
            if shape_rotated[x, y]!= Color.BLACK:
                output_grid[x, y] = shape_rotated[x, y]  # Only replace if the rotated pixel is not black

    return output_grid