from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel expansion, symmetry, pattern duplication

# description:
# In the input, you will see a small pattern of colored pixels. To create the output, 
# duplicate the pattern symmetrically across the grid, expanding it outward until it fills the entire grid, 
# while maintaining the original colors and spacing of the pixels.

def transform(input_grid):
    # Find the connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK)

    # Assuming there's only one main pattern in the input
    pattern = objects[0]
    # Crop the pattern to get the sprite
    sprite = crop(pattern, background=Color.BLACK)

    # Get the dimensions of the pattern
    pattern_height, pattern_width = sprite.shape

    # Determine the size of the output grid
    output_height = input_grid.shape[0]
    output_width = input_grid.shape[1]
    
    # Create the output grid filled with the background color
    output_grid = np.full((output_height, output_width), Color.BLACK)

    # Calculate the number of times to duplicate the pattern in both dimensions
    vertical_dups = output_height // pattern_height
    horizontal_dups = output_width // pattern_width

    # Fill the output grid with the expanded pattern
    for i in range(vertical_dups):
        for j in range(horizontal_dups):
            # Calculate the position to place the pattern
            x_pos = i * pattern_height
            y_pos = j * pattern_width
            
            # Blit the sprite into the output grid
            blit_sprite(output_grid, sprite, x=x_pos, y=y_pos, background=Color.BLACK)

    return output_grid