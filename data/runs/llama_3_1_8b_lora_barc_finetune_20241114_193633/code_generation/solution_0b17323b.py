from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry, reflection, color mapping

# description:
# In the input, you will see a grid with a single colored object (which can be of any color) on a black background.
# To create the output, reflect this object across the vertical axis of the grid and fill the newly created areas with the color of the original object.

def transform(input_grid):
    # Copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # Detect the object in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)

    # There should only be one object
    assert len(objects) == 1
    original_object = objects[0]

    # Get the color of the original object
    original_color = np.unique(original_object[original_object!= Color.BLACK])[0]

    # Reflect the object across the vertical axis
    reflected_object = np.flip(original_object, axis=1)

    # Find the center of the original object
    center_x, center_y = original_object.shape[0] // 2, original_object.shape[1] // 2

    # Calculate the position to place the reflected object
    # For the reflection, we need to calculate the new positions
    for x in range(original_object.shape[0]):
        for y in range(original_object.shape[1]):
            if original_object[x, y]!= Color.BLACK:
                # Calculate the new position for the reflected object
                reflected_x = x
                reflected_y = output_grid.shape[1] - 1 - y
                # Fill the reflected position with the original color
                if 0 <= reflected_x < output_grid.shape[0] and 0 <= reflected_y < output_grid.shape[1]:
                    output_grid[reflected_x, reflected_y] = original_color

    return output_grid