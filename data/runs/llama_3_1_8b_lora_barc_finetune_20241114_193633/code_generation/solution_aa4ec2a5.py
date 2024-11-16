from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel grouping, color mapping

# description:
# In the input, you will see a grid of colored pixels. The grid contains various colored pixels, some of which are blue, red, or yellow.
# To make the output grid, for each blue pixel, you should change its color to blue and replace it with a corresponding pixel of the same color in a specific position:
# - For blue pixels in the upper half of the grid, replace them with blue pixels in the lower half.
# - For red pixels in the lower half of the grid, replace them with red pixels in the upper half.
# - For yellow pixels in the left half of the grid, replace them with yellow pixels in the right half.
# - For yellow pixels in the right half of the grid, replace them with yellow pixels in the left half.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    height, width = input_grid.shape

    for x in range(height):
        for y in range(width):
            color = input_grid[x, y]
            if color == Color.BLUE:
                # Change color and swap with a pixel in the lower half
                if x < height // 2:
                    output_grid[x, y] = Color.BLUE
                    output_grid[height - 1 - x, y] = Color.BLUE
            elif color == Color.RED:
                # Change color and swap with a pixel in the upper half
                if x >= height // 2:
                    output_grid[x, y] = Color.RED
                    output_grid[height - 1 - x, y] = Color.RED
            elif color == Color.YELLOW:
                # Change color and swap with a pixel in the right half
                if y < width // 2:
                    output_grid[x, y] = Color.YELLOW
                    output_grid[x, width - 1 - y] = Color.YELLOW
                elif y >= width // 2:
                    output_grid[x, y] = Color.YELLOW
                    output_grid[x, width - 1 - y] = Color.YELLOW

    return output_grid