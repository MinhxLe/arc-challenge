from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel manipulation, symmetry, color replacement

# description:
# In the input, you will see a grid containing a colored shape made up of various colored pixels.
# To make the output, create a symmetrical version of the shape by reflecting it across the vertical axis,
# and then color the new symmetrical shape in a different color while leaving the original shape unchanged.

def transform(input_grid):
    # Create a copy of the input grid for the output
    output_grid = input_grid.copy()
    
    # Detect the colored objects in the input grid
    objects = detect_objects(grid=input_grid, colors=Color.NOT_BLACK, monochromatic=False, connectivity=4)
    
    for obj in objects:
        # Get the bounding box of the object
        x, y, w, h = bounding_box(obj)
        sprite = crop(obj)

        # Reflect the sprite across the vertical axis
        reflected_sprite = np.flip(sprite, axis=1)
        
        # Find the new position for the reflected sprite
        new_x, new_y = x, y + w
        
        # Place the reflected sprite into the output grid
        blit_sprite(output_grid, reflected_sprite, new_x, new_y, background=Color.BLACK)
        
        # Color the reflected sprite in a different color (for example, Color.GREEN)
        reflected_sprite[reflected_sprite!= Color.BLACK] = Color.GREEN

    return output_grid