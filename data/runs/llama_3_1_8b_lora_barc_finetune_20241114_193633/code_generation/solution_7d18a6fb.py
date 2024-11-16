from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color merging, grid merging

# description:
# In the input, you will see two overlapping grids with different colors. 
# To make the output, merge the two grids by replacing overlapping pixels with the color of the topmost grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Determine the height and width of the input grid
    height, width = input_grid.shape
    
    # Initialize the output grid with the background color
    output_grid = np.full((height, width), Color.BLACK)
    
    # Iterate through each pixel in the input grid
    for x in range(height):
        for y in range(width):
            color1 = input_grid[x, y]
            # If the pixel is not the background, copy it to the output grid
            if color1!= Color.BLACK:
                output_grid[x, y] = color1

    return output_grid