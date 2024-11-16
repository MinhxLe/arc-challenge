from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern recognition, tiling, symmetry

# description:
# In the input, you will see a grid containing a small repeating pattern of colored pixels. 
# To create the output, identify the pattern and replicate it throughout the entire grid, ensuring that the pattern remains aligned with the grid structure.

def transform(input_grid):
    # Find the connected components in the input grid
    components = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)

    # Assume there's only one pattern, get the first one
    pattern = components[0]
    
    # Get the bounding box of the pattern
    pattern_x, pattern_y, pattern_w, pattern_h = bounding_box(pattern, background=Color.BLACK)

    # Crop the pattern to get rid of the background
    pattern_cropped = crop(pattern, background=Color.BLACK)

    # Determine the size of the output grid
    output_height = (input_grid.shape[0] // pattern_h) * pattern_h
    output_width = (input_grid.shape[1] // pattern_w) * pattern_w
    output_grid = np.full((output_height, output_width), Color.BLACK)

    # Tile the cropped pattern across the output grid
    for i in range(0, output_height, pattern_h):
        for j in range(0, output_width, pattern_w):
            blit_sprite(output_grid, pattern_cropped, x=j, y=i, background=Color.BLACK)

    return output_grid