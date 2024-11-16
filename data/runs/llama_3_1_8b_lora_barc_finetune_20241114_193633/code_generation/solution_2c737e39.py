from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# shape detection, mirroring, grid transformation

# description:
# In the input, you will see a grid containing a colorful shape on a black background.
# To make the output, create a mirrored version of the shape on the opposite side of the grid, 
# ensuring that the colors and structure are preserved in the mirrored shape.

def transform(input_grid):
    # Detect the colorful shape in the input grid
    objects = detect_objects(grid=input_grid, monochromatic=False, connectivity=4, colors=Color.NOT_BLACK)
    assert len(objects) == 1, "There should be exactly one colorful shape in the input grid."
    shape = objects[0]

    # Get the bounding box of the shape
    x, y, width, height = bounding_box(shape)

    # Create the output grid
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Mirror the shape horizontally
    mirrored_shape = shape.copy()
    mirrored_shape = mirrored_shape[:, ::-1]

    # Calculate the position to place the mirrored shape
    mirrored_x, mirrored_y = x, y + height

    # Blit the original shape onto the output grid
    blit_sprite(output_grid, shape, x=x, y=y, background=Color.BLACK)

    # Blit the mirrored shape onto the output grid
    blit_sprite(output_grid, mirrored_shape, x=mirrored_x, y=mirrored_y, background=Color.BLACK)

    return output_grid