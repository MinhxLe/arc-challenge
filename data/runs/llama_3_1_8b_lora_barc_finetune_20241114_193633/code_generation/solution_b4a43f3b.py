from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# scaling, grid transformation

# description:
# In the input, you will see a grid with various colored objects, where each object has a distinct color.
# To make the output, scale each object in the grid by a factor of 2, effectively doubling its size 
# while maintaining the color and position relative to the grid.

def transform(input_grid):
    # Detect all the objects in the input grid
    objects = detect_objects(grid=input_grid, monochromatic=True, background=Color.BLACK, connectivity=4, allowed_dimensions=None, colors=None, can_overlap=False)
    
    # Initialize the output grid, which is double the size of the input grid
    output_grid = np.full((input_grid.shape[0] * 2, input_grid.shape[1] * 2), Color.BLACK)

    # Scale each detected object and place it in the output grid
    for obj in objects:
        # Get the bounding box of the object
        x, y, w, h = bounding_box(obj)
        
        # Crop the object to get its sprite
        sprite = crop(obj, background=Color.BLACK)

        # Scale the sprite by a factor of 2
        scaled_sprite = scale_pattern(sprite, scale_factor=2)

        # Calculate the position to place the scaled sprite in the output grid
        output_x = x * 2
        output_y = y * 2
        
        # Blit the scaled sprite onto the output grid
        blit_sprite(output_grid, scaled_sprite, x=output_x, y=output_y, background=Color.BLACK)

    return output_grid