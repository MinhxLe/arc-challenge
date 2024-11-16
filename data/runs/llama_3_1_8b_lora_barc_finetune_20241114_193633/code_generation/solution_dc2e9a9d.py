from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object extraction, symmetry, color filling

# description:
# In the input, you will see a grid containing a complex object made up of several colors.
# To create the output, you should extract the object from the grid and create a new grid that is a mirror image of the object.
# The output grid should be filled with the color of the object on one side and the background color on the other.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Detect the object in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)
    
    # Assuming there is only one object in the input
    if len(objects) == 0:
        return np.zeros_like(input_grid)  # Return an empty grid if no object found
    
    # Crop the first detected object
    object_sprite = crop(objects[0], background=Color.BLACK)
    
    # Create the output grid
    output_grid = np.full(input_grid.shape, Color.BLACK, dtype=int)

    # Determine the color of the object
    object_color = np.unique(object_sprite[object_sprite!= Color.BLACK])[0]

    # Calculate the dimensions of the object
    obj_height, obj_width = object_sprite.shape

    # Fill the output grid with the object color on one side and the background on the other
    for x in range(obj_height):
        for y in range(obj_width):
            if object_sprite[x, y]!= Color.BLACK:
                output_grid[x, y] = object_color
                # Calculate the mirrored position
                mirrored_x = x
                mirrored_y = output_grid.shape[1] - y - 1  # Mirror horizontally
                output_grid[mirrored_x, mirrored_y] = object_color

    return output_grid