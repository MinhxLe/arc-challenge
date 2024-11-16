from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object extraction, color transformation

# description:
# In the input, you will see a grid with several colored objects on a black background.
# To create the output, extract the largest object from the grid, transform its color to a specified color, and place it back into the grid in the same position.

def transform(input_grid):
    # Step 1: Extract the largest object from the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)
    
    # Find the largest object based on the number of pixels
    largest_object = max(objects, key=lambda obj: obj.size)
    
    # Step 2: Crop the largest object to get its sprite
    sprite = crop(largest_object, background=Color.BLACK)
    
    # Step 3: Create a new grid to place the transformed sprite back
    output_grid = np.full(input_grid.shape, Color.BLACK, dtype=int)
    
    # Step 4: Change the color of the largest object to a specified color
    new_color = Color.BLUE  # You can change this to any color you want
    sprite[sprite!= Color.BLACK] = new_color
    
    # Step 5: Place the transformed sprite back into the output grid
    blit_sprite(output_grid, sprite, x=0, y=0, background=Color.BLACK)
    
    return output_grid