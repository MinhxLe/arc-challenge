from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, boundary detection, color transformation

# description:
# The input grid contains a single object made of colored pixels surrounded by a black background. The object can be any shape but must be connected. 
# The task is to create an output grid where the color of the object is transformed based on its boundary:
# - The color of the object's boundary pixels (the pixels that are adjacent to the background) will change to the color of the object itself.
# - The inner pixels of the object will remain unchanged.

def transform(input_grid):
    # Create a copy of the input grid for the output
    output_grid = np.copy(input_grid)

    # Find the connected components in the grid (the object)
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)

    # Iterate over all objects found
    for obj in objects:
        # Get the bounding box of the object
        x, y, w, h = bounding_box(obj, background=Color.BLACK)

        # Get the color of the object
        object_color = obj[x, y]

        # Change the color of the boundary pixels to the object's color
        for i in range(x, x + w):
            for j in range(y, y + h):
                # Check if the current pixel is part of the boundary
                if (i == 0 or i == input_grid.shape[0] - 1 or j == 0 or j == input_grid.shape[1] - 1):
                    # Change the color of the boundary pixel to the object's color
                    output_grid[i, j] = object_color

    return output_grid