from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# rotation, color mapping, grid transformation

# description:
# In the input, you will see a grid containing a shape made of colored pixels on a black background.
# To create the output, rotate the shape 90 degrees clockwise and change its color based on a predefined mapping:
# blue -> red, red -> green, green -> blue, green -> yellow, yellow -> red, red -> blue, blue -> green.

def transform(input_grid):
    # Initialize output grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Define the color mapping
    color_map = {
        Color.BLUE: Color.RED,
        Color.RED: Color.GREEN,
        Color.GREEN: Color.BLUE,
        Color.GREEN: Color.YELLOW,
        Color.YELLOW: Color.RED,
        Color.RED: Color.BLUE,
        Color.BLUE: Color.GREEN
    }

    # Find the bounding box of the shape
    x, y, width, height = bounding_box(input_grid, background=Color.BLACK)

    # Extract the shape from the input grid
    shape = input_grid[x:x + width, y:y + height]

    # Rotate the shape 90 degrees clockwise
    rotated_shape = np.rot90(shape, k=-1)

    # Apply color mapping
    rotated_shape = np.vectorize(lambda color: color_map.get(color, color))(rotated_shape)

    # Place the rotated shape back into the output grid
    output_grid[x:x + rotated_shape.shape[0], y:y + rotated_shape.shape[1]] = rotated_shape

    return output_grid