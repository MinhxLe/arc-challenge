from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color transformation, boundary detection, pixel manipulation

# description:
# In the input, you will see a grid filled with colored pixels on a black background. 
# The goal is to create an output grid where:
# 1. For every colored pixel, if it has at least one neighboring pixel of the same color (up, down, left, or right), 
#    change its color to a new color (e.g., blue). 
# 2. If a colored pixel does not have any neighboring pixels of the same color, change its color to another color (e.g., green).

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    # Define new colors for transformations
    color_blue = Color.BLUE
    color_green = Color.GREEN

    # Get the dimensions of the grid
    rows, cols = input_grid.shape

    # Iterate through each pixel in the grid
    for x in range(rows):
        for y in range(cols):
            # Check the color of the current pixel
            current_color = input_grid[x, y]

            # Check neighbors (up, down, left, right)
            neighbors = [
                (x - 1, y),  # up
                (x + 1, y),  # down
                (x, y - 1),  # left
                (x, y + 1),  # right
            ]
            has_same_color_neighbor = any(
                0 <= nx < rows and 0 <= ny < cols and input_grid[nx, ny] == current_color
                for nx, ny in neighbors
            )

            # Determine the new color based on the neighboring condition
            if has_same_color_neighbor:
                output_grid[x, y] = color_blue
            else:
                output_grid[x, y] = color_green

    return output_grid