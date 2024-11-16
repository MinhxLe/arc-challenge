from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel rotation, color transformation

# description:
# In the input, you will see a grid containing a colored object represented by a collection of colored pixels.
# To create the output, rotate the object 90 degrees clockwise and change the color of the rotated pixels to a specified new color.
# The background remains unchanged.

def transform(input_grid):
    # Create an output grid initialized to the input grid
    output_grid = np.copy(input_grid)

    # Find the connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=False)

    # For each object found, rotate it 90 degrees clockwise
    for obj in objects:
        # Get the bounding box of the object
        x, y, width, height = bounding_box(obj)
        sprite = crop(obj)

        # Rotate the sprite 90 degrees clockwise
        rotated_sprite = np.rot90(sprite, k=-1)

        # Change the color of the rotated sprite to a new specified color (for example, Color.ORANGE)
        rotated_sprite[rotated_sprite!= Color.BLACK] = Color.ORANGE

        # Blit the rotated sprite back to the output grid
        blit_sprite(output_grid, rotated_sprite, x=x, y=y, background=Color.BLACK)

    return output_grid