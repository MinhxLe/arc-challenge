from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object rotation, color propagation, symmetry

# description:
# In the input, you will see a colored object on a black background. The object is a connected component of colored pixels.
# To create the output, rotate the object 90 degrees clockwise and propagate its color to the surrounding pixels in the grid.

def transform(input_grid):
    # Step 1: Find the colored object
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)
    assert len(objects) == 1  # There should be exactly one object
    object_pixels = objects[0]

    # Step 2: Create a blank output grid
    output_grid = np.zeros_like(input_grid)

    # Step 3: Rotate the object 90 degrees clockwise
    # Create an empty grid for the rotated object
    rotated_object = np.rot90(object_pixels, k=-1)

    # Step 4: Blit the rotated object onto the output grid
    blit_object(output_grid, rotated_object, background=Color.BLACK)

    # Step 5: Propagate the color of the rotated object to the surrounding pixels
    color = np.unique(rotated_object[rotated_object!= Color.BLACK])[0]  # Get the color of the object
    for x, y in np.argwhere(rotated_object!= Color.BLACK):
        # Check the 4-way connectivity neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < output_grid.shape[0] and 0 <= ny < output_grid.shape[1]:
                output_grid[nx, ny] = color

    return output_grid