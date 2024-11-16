from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, vertical and horizontal stacking

# description:
# In the input, you will see a grid containing colored shapes arranged vertically and horizontally. 
# Each shape has a unique color. To create the output, stack the shapes horizontally, maintaining their order, 
# and change their colors according to a specific mapping (e.g., red to blue, green to yellow, etc.).

def transform(input_grid):
    # Initialize output grid
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Detect the colored shapes in the input grid
    objects = detect_objects(grid=input_grid, background=Color.BLACK, monochromatic=False, connectivity=4)

    # Create a mapping for color transformation
    color_map = {
        Color.RED: Color.BLUE,
        Color.GREEN: Color.YELLOW,
        Color.YELLOW: Color.RED,
        Color.BLUE: Color.GREEN,
        Color.PURPLE: Color.ORANGE,
        Color.ORANGE: Color.PURPLE,
        Color.PINK: Color.BROWN,
        Color.BROWN: Color.PINK,
        Color.GRAY: Color.GRAY,
        Color.BLACK: Color.BLACK,
    }

    # Position for stacking
    x_offset = 0

    # Iterate over detected objects
    for obj in objects:
        # Crop the object to get its sprite
        sprite = crop(obj, background=Color.BLACK)

        # Get the color of the current object
        color = np.unique(sprite[sprite!= Color.BLACK])[0]

        # Change the color according to the mapping
        new_color = color_map[color]

        # Change the color of the sprite
        colored_sprite = np.where(sprite == color, new_color, Color.BLACK)

        # Blit the transformed sprite onto the output grid
        blit_sprite(output_grid, colored_sprite, x=x_offset, y=0)

        # Move to the right for the next sprite
        x_offset += colored_sprite.shape[1]

    return output_grid