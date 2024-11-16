from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# rotation, color propagation, filling

# description:
# In the input, you will see a grid with a central colored object and a black background.
# To make the output, rotate the object by 90 degrees clockwise and fill in the newly exposed edges
# with the original object's color.

def transform(input_grid):
    # Find the central object in the input grid
    objects = find_connected_components(input_grid, monochromatic=True, connectivity=4)
    
    # Assuming there is only one object
    assert len(objects) == 1
    original_object = objects[0]
    
    # Get the color of the original object
    object_color = np.unique(original_object[original_object!= Color.BLACK])[0]
    
    # Create an empty output grid
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Get the bounding box of the original object
    x, y, width, height = bounding_box(original_object)

    # Rotate the object 90 degrees clockwise
    rotated_object = np.rot90(original_object, k=-1)

    # Find the top-left corner of the original object to place the rotated object
    new_x, new_y = x + height, y + width

    # Fill the output grid with the rotated object
    blit_sprite(output_grid, rotated_object, new_x, new_y, background=Color.BLACK)

    # Fill in the newly exposed edges with the original object's color
    for i in range(output_grid.shape[0]):
        for j in range(output_grid.shape[1]):
            if output_grid[i, j] == Color.BLACK:
                # Check if this pixel is on the edge of the rotated object
                if (i < height and j < width) or (i >= height and i < 2 * height - 1 and j < width):
                    output_grid[i, j] = object_color

    return output_grid