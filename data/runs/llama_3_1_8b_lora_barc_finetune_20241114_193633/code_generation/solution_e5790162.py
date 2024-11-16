from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# filling, object detection, color transformation

# description:
# In the input, you will see a grid with a distinct colored shape and several black pixels scattered throughout.
# To make the output, fill in the black pixels with the color of the shape if they are adjacent to it.

def transform(input_grid):
    # Create an output grid initialized with the input grid
    output_grid = np.copy(input_grid)

    # Find the color of the distinct shape (non-black pixels)
    shape_color = None
    for x, y in np.argwhere(input_grid!= Color.BLACK):
        shape_color = input_grid[x, y]
        break

    # Fill adjacent black pixels with the shape's color
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            if input_grid[x, y] == Color.BLACK:
                # Check if this pixel is adjacent to the shape
                if (x > 0 and input_grid[x - 1, y] == shape_color) or \
                   (x < input_grid.shape[0] - 1 and input_grid[x + 1, y] == shape_color) or \
                   (y > 0 and input_grid[x, y - 1] == shape_color) or \
                   (y < input_grid.shape[1] - 1 and input_grid[x, y + 1] == shape_color):
                    output_grid[x, y] = shape_color

    return output_grid