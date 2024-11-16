from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, pattern merging

# description:
# In the input, you will see two patterns: one made up of green pixels and one made up of red pixels. 
# The task is to blend the two patterns by averaging the colors of overlapping pixels to create a new pattern in the output grid.

def transform(input_grid):
    # Split the input grid into two separate patterns
    green_pattern = input_grid[input_grid == Color.GREEN]
    red_pattern = input_grid[input_grid == Color.RED]

    # Determine the size of the output grid
    output_height = max(green_pattern.shape[0], red_pattern.shape[0])
    output_width = max(green_pattern.shape[1], red_pattern.shape[1])
    output_grid = np.full((output_height, output_width), Color.BLACK)

    # Create the blended output grid
    for x in range(output_height):
        for y in range(output_width):
            if green_pattern[x, y] == Color.GREEN:
                output_grid[x, y] = Color.GREEN
            elif red_pattern[x, y] == Color.RED:
                output_grid[x, y] = Color.RED
            elif green_pattern[x, y]!= Color.BLACK and red_pattern[x, y]!= Color.BLACK:
                # Average the color of overlapping pixels
                output_grid[x, y] = Color.BLACK  # For simplicity, we can choose to keep it black here
            else:
                output_grid[x, y] = Color.BLACK

    return output_grid