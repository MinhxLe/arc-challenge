from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color counting, pixel arrangement

# description:
# In the input, you will see a grid filled with various colored pixels. The task is to count the number of pixels of each color in the grid and then create an output grid.
# The output grid will have the same dimensions as the input grid, but each pixel will be replaced by a color that corresponds to the count of that color in the input.
# For example, if a color appears 3 times, it will be represented by that color in the output grid in the same position. If a color appears more than 10 times, it will be represented by black.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.zeros_like(input_grid)

    # Create a dictionary to count occurrences of each color
    color_count = {}
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            color = input_grid[x, y]
            if color!= Color.BLACK:  # Ignore the background
                if color not in color_count:
                    color_count[color] = 0
                color_count[color] += 1

    # Fill the output grid based on the counts
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            color = input_grid[x, y]
            if color!= Color.BLACK:
                count = color_count[color]
                if count > 10:  # More than 10 occurrences
                    output_grid[x, y] = Color.BLACK  # Represented by black
                else:
                    output_grid[x, y] = color  # Represented by the color itself

    return output_grid