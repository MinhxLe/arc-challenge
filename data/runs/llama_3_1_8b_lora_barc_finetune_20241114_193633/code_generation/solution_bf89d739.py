from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, reflection, bounding box

# description:
# In the input, you will see a grid with a few colored pixels. 
# To make the output grid, reflect the colors of the pixels across the vertical axis of the grid. 
# If a pixel is located at (x, y), it will be reflected to (n-1-x, y), where n is the width of the grid.
# The output grid should maintain the same shape as the input grid.

def transform(input_grid):
    # Get the shape of the input grid
    n, m = input_grid.shape
    
    # Create an output grid initialized to the background color
    output_grid = np.full((n, m), Color.BLACK)
    
    # Iterate through each pixel in the input grid
    for x in range(n):
        for y in range(m):
            if input_grid[x, y]!= Color.BLACK:  # Only consider colored pixels
                reflected_x = n - 1 - x  # Reflect across the vertical axis
                output_grid[reflected_x, y] = input_grid[x, y]  # Set the reflected pixel
                
    return output_grid