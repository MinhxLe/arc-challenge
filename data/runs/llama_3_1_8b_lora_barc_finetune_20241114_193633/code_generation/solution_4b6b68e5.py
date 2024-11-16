from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, pixel manipulation

# description:
# In the input grid, you will see a collection of colored pixels on a black background. 
# The task is to create a new grid where each pixel of the original grid is transformed according to the following rules:
# 1. For each pixel, if it is not black, if it is surrounded by pixels of the same color, change it to that color.
# 2. If a pixel is surrounded by different colors, change it to the color of the majority of its neighbors (in case of a tie, choose the first color in the order of defined colors).
# 3. Pixels that are isolated (i.e., have no neighboring pixels of the same color) will remain unchanged.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    # Get the dimensions of the grid
    rows, cols = input_grid.shape

    for x in range(rows):
        for y in range(cols):
            # Get the color of the current pixel
            current_color = input_grid[x, y]

            # Check if the current pixel is not black
            if current_color!= Color.BLACK:
                # Check the neighboring pixels (4-way connectivity)
                neighbors = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and input_grid[nx, ny] == current_color:
                        neighbors.append(input_grid[nx, ny])

                # If there are neighbors of the same color, change to that color
                if neighbors:
                    unique_colors = np.unique(neighbors)
                    if Color.BLACK not in unique_colors:
                        output_grid[x, y] = unique_colors[0]

    return output_grid