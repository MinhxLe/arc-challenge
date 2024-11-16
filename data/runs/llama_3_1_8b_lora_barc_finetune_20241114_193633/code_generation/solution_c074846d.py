from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# rotation, color transformation

# description:
# In the input, you will see a colored shape that is symmetric along the center. 
# To create the output, rotate the shape 90 degrees clockwise and change its color to a specified new color.

def transform(input_grid):
    # Get the bounding box of the non-background pixels
    x, y, width, height = bounding_box(input_grid, background=Color.BLACK)
    sprite = input_grid[x:x + width, y:y + height]

    # Rotate the sprite 90 degrees clockwise
    rotated_sprite = np.rot90(sprite, k=-1)

    # Create the output grid
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Change the color of the rotated sprite to a new color
    new_color = Color.RED  # You can change this to any color you want
    rotated_sprite[rotated_sprite!= Color.BLACK] = new_color

    # Blit the rotated sprite onto the output grid
    blit_sprite(output_grid, rotated_sprite, background=Color.BLACK)

    return output_grid