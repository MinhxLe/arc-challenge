from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# rotation, color transformation

# description:
# In the input, you will see a grid with a colored object in the center and a gray border surrounding it.
# To make the output, rotate the object by 90 degrees clockwise and fill the new position with the color of the original object.
# The output grid should maintain the same size as the input grid.

def transform(input_grid):
    # Create an output grid with the same shape as the input grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Find the bounding box of the non-background pixels in the input grid
    x, y, width, height = bounding_box(input_grid, background=Color.BLACK)

    # Extract the object from the input grid
    object_region = input_grid[x:x + width, y:y + height]

    # Rotate the object by 90 degrees clockwise
    rotated_object = np.rot90(object_region, k=-1)

    # Calculate the position to place the rotated object
    output_x = x
    output_y = y

    # Determine the new position to blit the rotated object
    output_grid = blit_sprite(output_grid, rotated_object, x=output_x, y=output_y, background=Color.BLACK)

    # Get the color of the original object
    original_color = object_region[object_region!= Color.BLACK][0]

    # Fill the new position with the original color
    output_grid[output_grid!= Color.BLACK] = original_color

    return output_grid