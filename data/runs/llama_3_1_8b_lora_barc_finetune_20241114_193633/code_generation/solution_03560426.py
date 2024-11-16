from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# rotation, color mapping, grid transformation

# description:
# In the input, you will see a grid with several colored shapes (which can be of different colors).
# To make the output, you should rotate each shape 90 degrees clockwise and place it back in the grid in the same position.
# If a shape overlaps with another shape, it should not change its color, but if it overlaps with the background, it should remain unchanged.

def transform(input_grid):
    # Find all connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=False)

    # Create an output grid initialized to the background color
    output_grid = np.full_like(input_grid, Color.BLACK)

    for obj in objects:
        # Get the bounding box of the current object
        x, y, width, height = bounding_box(obj, background=Color.BLACK)
        sprite = crop(obj, background=Color.BLACK)

        # Rotate the sprite 90 degrees clockwise
        rotated_sprite = np.rot90(sprite, k=-1)  # k=-1 for clockwise rotation

        # Find the position to place the rotated sprite in the output grid
        blit_sprite(output_grid, rotated_sprite, x=x, y=y, background=Color.BLACK)

    return output_grid