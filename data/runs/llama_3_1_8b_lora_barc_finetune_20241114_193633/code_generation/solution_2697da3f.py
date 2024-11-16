from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# radial symmetry, object duplication, color mapping

# description:
# In the input, you will see a colored object in the center of a grid. The task is to create a new grid that duplicates the object around the center point,
# replicating it in a radial pattern (4-way symmetry) while ensuring that the original object remains in the center.
# The output grid should be the smallest possible size that contains all duplicated objects.

def transform(input_grid):
    # Step 1: Detect the central object in the grid
    objects = detect_objects(grid=input_grid, monochromatic=False, connectivity=8, colors=[Color.NOT_BLACK], allowed_dimensions=None)
    
    # We assume there's only one object in the center
    assert len(objects) == 1
    central_object = objects[0]
    
    # Step 2: Crop the central object to remove any surrounding background
    sprite = crop(central_object, background=Color.BLACK)
    sprite_height, sprite_width = sprite.shape

    # Step 3: Create the output grid
    output_size = max(sprite_height, sprite_width) * 2 + 1
    output_grid = np.full((output_size, output_size), Color.BLACK)

    # Step 4: Blit the central object at the center of the output grid
    center_x = output_size // 2
    center_y = output_size // 2
    blit_sprite(output_grid, sprite, x=center_x - sprite_width // 2, y=center_y - sprite_height // 2, background=Color.BLACK)

    # Step 5: Duplicate the object in a radial pattern
    for i in range(4):
        for j in range(4):
            # Calculate the positions for the four quadrants
            blit_sprite(output_grid, sprite, x=center_x + i * (sprite_width // 2), y=center_y + j * (sprite_height // 2), background=Color.BLACK)

    return output_grid