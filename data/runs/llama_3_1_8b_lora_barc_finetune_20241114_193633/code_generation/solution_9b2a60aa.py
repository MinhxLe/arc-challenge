from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# reflection, color propagation

# description:
# In the input, you will see a grid with colored pixels on a black background. The goal is to reflect the colors of the pixels across the grid's center. 
# The colors that are reflected should be placed in their new positions, while any new pixels that would overlap with existing colored pixels should be filled with the color of the existing pixel.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Get the dimensions of the input grid
    n, m = input_grid.shape
    output_grid = np.copy(input_grid)

    # Calculate the center of the grid
    center_x, center_y = n // 2, m // 2

    # Reflect colors across the center
    for x in range(n):
        for y in range(m):
            if input_grid[x, y]!= Color.BLACK:
                # Calculate the reflected positions
                reflected_x = (2 * center_x) - x
                reflected_y = (2 * center_y) - y
                
                # Ensure the reflected positions are within bounds
                if 0 <= reflected_x < n and 0 <= reflected_y < m:
                    # If the reflected position is already occupied by a color, keep it
                    if output_grid[reflected_x, reflected_y]!= Color.BLACK:
                        continue
                    # Otherwise, place the original color in the reflected position
                    output_grid[reflected_x, reflected_y] = input_grid[x, y]

    return output_grid