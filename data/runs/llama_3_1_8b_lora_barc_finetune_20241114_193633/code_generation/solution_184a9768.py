from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object detection, color mapping

# description:
# In the input, you will see a grid with various colored objects, some of which are overlapping. 
# The output should change the color of each object to the color of the object that is directly above it in the grid, if it exists. 
# If there is no object directly above, the color remains unchanged.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)
    height, width = input_grid.shape

    # Iterate through each pixel in the grid
    for x in range(height):
        for y in range(width):
            # Check if the current pixel is not black (background)
            if input_grid[x, y]!= Color.BLACK:
                # Check for the pixel above it
                if x > 0 and input_grid[x - 1, y]!= Color.BLACK:
                    output_grid[x, y] = input_grid[x - 1, y]  # Change color to the one above
                elif x == 0:  # If it's the top row, it remains the same
                    output_grid[x, y] = input_grid[x, y]

    return output_grid