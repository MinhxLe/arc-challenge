from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# repetition, tiling

# description:
# In the input, you will see a small repeating pattern of colored pixels on a black background. 
# To create the output, replicate this pattern multiple times across a larger grid, ensuring that the patterns do not overlap.

def transform(input_grid):
    # Find the connected components (the patterns) in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True, connectivity=4)
    
    # Assuming there is only one pattern, extract it
    pattern = crop(objects[0], background=Color.BLACK)

    # Determine the size of the output grid
    output_size = (input_grid.shape[0] * 3, input_grid.shape[1] * 3)  # Repeat the pattern 3x
    output_grid = np.full(output_size, Color.BLACK)

    # Calculate the position to place the pattern in the output grid
    for i in range(0, output_size[0], pattern.shape[0]):
        for j in range(0, output_size[1], pattern.shape[1]):
            # Blit the pattern onto the output grid
            blit_sprite(output_grid, pattern, x=i, y=j, background=Color.BLACK)

    return output_grid