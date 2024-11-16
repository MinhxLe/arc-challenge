from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, shape transformation

# description:
# In the input, you will see a grid with colored shapes. 
# To make the output, each shape should be transformed by replacing its color with the color of the pixel in the same position in the input grid, 
# and shifting the shape right and down by one pixel. If a shape goes out of bounds, it wraps around to the other side of the grid.

def transform(input_grid):
    # Initialize the output grid with the same shape as the input grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Find all the connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK)

    for obj in objects:
        # Get the bounding box of each object
        x, y, width, height = bounding_box(obj, background=Color.BLACK)

        # Get the color of the object
        color = obj[x, y]

        # Create a shifted version of the object
        for i in range(height):
            for j in range(width):
                if obj[i, j]!= Color.BLACK:
                    # Calculate new position
                    new_x = (x + 1) % output_grid.shape[0]
                    new_y = (y + 1) % output_grid.shape[1]
                    output_grid[new_x, new_y] = input_grid[x, y]  # Use the color from the original position

    return output_grid