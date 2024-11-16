from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel transformation, tiling, pattern repetition

# description:
# In the input, you will see a small pattern of colored pixels in the top left corner of the grid. 
# To create the output, tile the entire grid with this pattern, repeating it across the entire grid, 
# ensuring that the pattern is not overlapping and fills the entire grid completely.

def transform(input_grid):
    # Get the bounding box of the pattern
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True, connectivity=4)
    assert len(objects) == 1, "Expected exactly one pattern in the input grid."
    
    pattern = objects[0]
    pattern_bbox = bounding_box(pattern, background=Color.BLACK)
    pattern_height, pattern_width = pattern_bbox[3], pattern_bbox[2]

    # Determine the size of the output grid
    output_height = (input_grid.shape[0] // pattern_height) * pattern_height
    output_width = (input_grid.shape[1] // pattern_width) * pattern_width

    # Create the output grid initialized to the background color
    output_grid = np.full((output_height, output_width), Color.BLACK)

    # Tile the pattern across the output grid
    for i in range(0, output_height, pattern_height):
        for j in range(0, output_width, pattern_width):
            # Blit the pattern onto the output grid
            blit_sprite(output_grid, pattern, x=j, y=i, background=Color.BLACK)

    return output_grid