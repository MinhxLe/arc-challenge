from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# rotation, color mapping, tiling

# description:
# In the input, you will see a small colored sprite (a pattern) surrounded by a larger grid of black pixels. 
# To make the output, rotate the sprite 90 degrees clockwise and tile it across the grid. 
# Each rotation should be represented by a distinct color, cycling through a predefined set of colors.

def transform(input_grid):
    # Define the color mapping for each rotation
    rotation_colors = [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW, Color.ORANGE]
    
    # Extract the sprite from the input grid
    sprite = crop(input_grid, background=Color.BLACK)
    sprite_height, sprite_width = sprite.shape

    # Create an output grid of the appropriate size
    output_grid_height = input_grid.shape[0]
    output_grid_width = input_grid.shape[1]
    output_grid = np.full((output_grid_height, output_grid_width), Color.BLACK)

    # Tile the rotated sprites across the output grid
    for i in range(0, output_grid_height, sprite_height):
        for j in range(0, output_grid_width, sprite_width):
            # Determine the rotation count
            rotation_count = (i // sprite_height + j // sprite_width) % len(rotation_colors)
            # Rotate the sprite and blit onto the output grid
            rotated_sprite = np.rot90(sprite, k=-rotation_count)  # Rotate 90 degrees clockwise
            blit_sprite(output_grid, rotated_sprite, x=j, y=i, background=Color.BLACK)

    return output_grid