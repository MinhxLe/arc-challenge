from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color counting, boundary detection, symmetry

# description:
# In the input, you will see a grid filled with various colored objects. The task is to identify the largest object 
# based on color and create a new grid that reflects that object symmetrically across the vertical axis, 
# filling the background with a contrasting color.

def transform(input_grid):
    # 1. Find the largest object in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=8, monochromatic=False)

    # 2. Determine the largest object by size
    largest_object = max(objects, key=lambda obj: obj.size, default=None)

    # 3. Create an output grid
    if largest_object is not None:
        output_grid = np.full(largest_object.shape, Color.BLACK)
        # Blit the largest object into the output grid
        blit_object(output_grid, largest_object, background=Color.BLACK)
    else:
        output_grid = np.full(input_grid.shape, Color.BLACK)

    # 4. Reflect the largest object across the vertical axis
    if largest_object is not None:
        # Crop the largest object to avoid extra black pixels
        sprite = crop(largest_object, background=Color.BLACK)
        # Reflect the object horizontally
        reflected_sprite = np.flip(sprite, axis=1)
        # Find the position to place the reflected sprite
        x_offset = (output_grid.shape[1] - reflected_sprite.shape[1]) // 2
        y_offset = (output_grid.shape[0] - reflected_sprite.shape[0]) // 2
        blit_sprite(output_grid, reflected_sprite, x_offset, y_offset, background=Color.BLACK)

    return output_grid