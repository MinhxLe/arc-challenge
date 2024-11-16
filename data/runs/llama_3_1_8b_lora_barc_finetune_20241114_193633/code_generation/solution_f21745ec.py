from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object detection, scaling, color replacement

# description:
# In the input, you will see several colored objects on a black background. 
# To make the output, detect all objects of a specific color and scale them to twice their original size, 
# while replacing the original colors with a specified new color.

def transform(input_grid):
    # Detect all objects in the grid of a specific color
    objects = detect_objects(grid=input_grid, colors=[Color.BLUE], monochromatic=True, connectivity=4, allowed_dimensions=[(1, 1)])

    # Create an output grid initialized to the background color
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Scale factor for scaling the objects
    scale_factor = 2

    # New color to use for the scaled objects
    new_color = Color.RED

    # For each detected object, scale it and blit it onto the output grid
    for obj in objects:
        # Crop the object to remove any extra black pixels around it
        cropped_obj = crop(obj, background=Color.BLACK)

        # Scale the object to twice its size
        scaled_object = scale_pattern(cropped_obj, scale_factor)

        # Find the position of the original object to place the scaled object
        x, y = np.argwhere(cropped_obj!= Color.BLACK)[0]  # Get the coordinates of the first non-black pixel

        # Blit the scaled object onto the output grid
        blit_sprite(output_grid, scaled_object, x, y, background=Color.BLACK)

    return output_grid