from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern replication, grid transformation

# description:
# In the input, you will see a small repeating pattern of colored pixels. 
# To make the output, replicate this pattern across a larger grid, ensuring that the new pattern is aligned properly and fills the entire grid.

def transform(input_grid):
    # Detect the pattern by finding connected components
    patterns = find_connected_components(input_grid, background=Color.BLACK, connectivity=4)

    # Assuming there is exactly one pattern in the input
    assert len(patterns) == 1
    pattern = patterns[0]
    
    # Get the bounding box of the pattern
    x, y, w, h = bounding_box(pattern)
    
    # Crop the pattern to remove any background
    cropped_pattern = crop(pattern, background=Color.BLACK)

    # Determine the size of the output grid
    output_height = (input_grid.shape[0] // h) * h
    output_width = (input_grid.shape[1] // w) * w
    output_grid = np.full((output_height, output_width), Color.BLACK)

    # Fill the output grid with the pattern
    for i in range(0, output_height, h):
        for j in range(0, output_width, w):
            blit_sprite(output_grid, cropped_pattern, x=i, y=j, background=Color.BLACK)

    return output_grid