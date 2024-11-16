from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern recognition, tiling, color filling

# description:
# In the input, you will see a pattern of colored pixels arranged in a rectangular shape. 
# To create the output, replicate this pattern in a grid format, filling the entire grid with the pattern while ensuring that the original pattern is maintained.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Identify the bounding box of the original pattern
    bounding_box_coords = bounding_box(input_grid, background=Color.BLACK)
    x, y, width, height = bounding_box_coords

    # Extract the pattern from the input grid
    pattern = input_grid[x:x + width, y:y + height]

    # Determine the size of the output grid
    output_height = (input_grid.shape[0] // height) * height
    output_width = (input_grid.shape[1] // width) * width

    # Create the output grid filled with the background color
    output_grid = np.full((output_height, output_width), Color.BLACK)

    # Fill the output grid with the pattern
    for i in range(0, output_height, height):
        for j in range(0, output_width, width):
            # Blit the pattern into the output grid
            blit_sprite(output_grid, pattern, x=j, y=i, background=Color.BLACK)

    return output_grid