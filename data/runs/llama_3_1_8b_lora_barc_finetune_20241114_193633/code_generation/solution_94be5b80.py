from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel shifting, boundary detection

# description:
# In the input, you will see a grid with colored pixels and a black background. 
# To create the output, shift each colored pixel towards the nearest edge of the grid 
# in the direction of the nearest edge pixel. If a pixel is already at the edge, it remains unchanged.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)
    height, width = input_grid.shape

    for x in range(height):
        for y in range(width):
            if input_grid[x, y]!= Color.BLACK:
                # Find the nearest edge
                nearest_edge_x = x
                nearest_edge_y = y
                
                # Check top edge
                if x == 0:
                    nearest_edge_x = x
                # Check bottom edge
                elif x == height - 1:
                    nearest_edge_x = x
                # Check left edge
                if y == 0:
                    nearest_edge_y = y
                # Check right edge
                elif y == width - 1:
                    nearest_edge_y = y
                
                # Calculate the shift direction
                shift_x = nearest_edge_x - x
                shift_y = nearest_edge_y - y
                
                # Shift the pixel towards the nearest edge
                if 0 <= x + shift_x < height and 0 <= y + shift_y < width:
                    output_grid[x + shift_x, y + shift_y] = input_grid[x, y]
                else:
                    # If out of bounds, keep the original color
                    output_grid[x, y] = input_grid[x, y]

    return output_grid