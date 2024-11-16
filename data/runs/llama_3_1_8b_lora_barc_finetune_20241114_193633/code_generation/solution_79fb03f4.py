from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel manipulation, color transformation, diagonal lines

# description:
# In the input grid, you will see a pattern of colored pixels that may include a few blue pixels. 
# To create the output, draw a line of a specified color (e.g., blue) from each blue pixel to the nearest horizontal or vertical edge of the grid, 
# creating a boundary effect.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)
    height, width = input_grid.shape

    # Iterate over each pixel in the grid
    for x in range(height):
        for y in range(width):
            if input_grid[x, y] == Color.BLUE:
                # Check horizontal boundary
                for dx in range(1, width):
                    if output_grid[x, y + dx] == Color.BLACK:
                        output_grid[x, y + dx] = Color.BLUE
                    if output_grid[x, y - dx] == Color.BLACK:
                        output_grid[x, y - dx] = Color.BLUE

                # Check vertical boundary
                for dy in range(1, height):
                    if output_grid[x + dy, y] == Color.BLUE:
                        output_grid[x + dy, y] = Color.BLUE
                    if output_grid[x - dy, y] == Color.BLUE:
                        output_grid[x - dy, y] = Color.BLUE

    return output_grid