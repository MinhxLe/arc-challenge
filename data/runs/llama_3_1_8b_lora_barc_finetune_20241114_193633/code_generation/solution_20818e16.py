from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# shape extraction, resizing

# description:
# In the input, you will see a grid containing a colored object surrounded by a black border. 
# To create the output, extract the colored object and resize it to fit in a 3x3 area in the center of a new grid.

def transform(input_grid):
    # Step 1: Crop the colored object from the input grid
    object_mask = (input_grid!= Color.BLACK)
    object_pixels = np.where(object_mask)
    
    # Crop the object from the input grid
    object_sprite = crop(input_grid[object_mask], background=Color.BLACK)

    # Step 2: Resize the cropped object to fit into a 3x3 grid
    output_grid = np.full((3, 3), Color.BLACK, dtype=int)

    # Determine the center of the output grid
    center_x, center_y = 1, 1  # Center of a 3x3 grid

    # Resize the object to fit into the output grid
    for i in range(object_sprite.shape[0]):
        for j in range(object_sprite.shape[1]):
            if object_sprite[i, j]!= Color.BLACK:
                # Calculate the new position in the 3x3 grid
                new_x = center_x + (i - 1)  # Shift down by 1 to center
                new_y = center_y + (j - 1)  # Shift left by 1 to center
                output_grid[new_x, new_y] = object_sprite[i, j]

    return output_grid