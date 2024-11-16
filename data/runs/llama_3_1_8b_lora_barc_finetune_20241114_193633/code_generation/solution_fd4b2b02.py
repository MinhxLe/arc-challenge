from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry, reflection, color transformation

# description:
# In the input, you will see a grid with a central object and various colored pixels around it.
# To make the output, reflect the object across the vertical and horizontal axes of the grid, creating a symmetrical pattern.
# The output grid should contain the original object and its reflections.

def transform(input_grid):
    # Create a copy of the input grid to hold the output
    output_grid = np.copy(input_grid)

    # Detect the central object in the grid
    objects = detect_objects(input_grid, background=Color.BLACK, monochromatic=False, connectivity=4)
    
    # There should be only one object detected
    assert len(objects) == 1
    central_object = objects[0]

    # Get the bounding box of the central object
    x, y, width, height = bounding_box(central_object, background=Color.BLACK)

    # Get the dimensions of the original object
    original_width = width
    original_height = height

    # Create the reflections
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        reflected_object = np.flipud(np.fliplr(central_object))  # Reflect both horizontally and vertically
        blit_sprite(output_grid, reflected_object, x + dx * original_width, y + dy * original_height, background=Color.BLACK)

    # Blit the original object to the output grid
    blit_sprite(output_grid, central_object, x, y, background=Color.BLACK)

    return output_grid