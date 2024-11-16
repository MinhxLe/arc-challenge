from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object detection, resizing, scaling, cropping, color filling

# description:
# In the input, you will see several objects on a grid of different colors, with a black background.
# To create the output, scale each object down by half its size and fill the newly exposed pixels with the color of the original object's background.

def transform(input_grid):
    # Detect all objects in the input grid
    objects = detect_objects(grid=input_grid, colors=Color.NOT_BLACK, monochromatic=False, connectivity=4)
    
    # Create an output grid initialized to the background color
    output_grid = np.full_like(input_grid, Color.BLACK)

    for obj in objects:
        # Crop the object to remove the background
        cropped_object = crop(obj, background=Color.BLACK)

        # Scale down the object by half its size
        scaled_object = np.zeros((cropped_object.shape[0] // 2, cropped_object.shape[1] // 2), dtype=int)
        for i in range(scaled_object.shape[0]):
            for j in range(scaled_object.shape[1]):
                scaled_object[i, j] = cropped_object[2 * i, 2 * j]  # Copy every second pixel

        # Calculate the position to place the scaled object back into the output grid
        x_offset = (x - obj.shape[0] // 2)  # Centering the scaled object
        y_offset = (y - obj.shape[1] // 2)
        
        # Place the scaled object into the output grid
        blit_sprite(output_grid, scaled_object, x=x_offset, y=y_offset, background=Color.BLACK)

    return output_grid