from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel counting, color transformation

# description:
# In the input, you will see a grid with colored pixels on a black background. The grid contains various colors and at least one color that forms a shape.
# To create the output grid, count the number of pixels of each color in the shape and transform the shape as follows:
# 1. For each color, if the count is greater than 3, change the color to the next color in the predefined color list.
# 2. If the count is less than or equal to 3, change the color to black.

def transform(input_grid):
    # Define the color list
    color_list = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GRAY, Color.PINK, Color.ORANGE, Color.PURPLE, Color.BROWN, Color.GRAY]
    output_grid = np.copy(input_grid)

    # Count the number of pixels of each color
    for color in Color.NOT_BLACK:
        count = np.sum(input_grid == color)
        # If the count is greater than 3, change to the next color in the list
        if count > 3 and count < len(color_list):
            output_grid[input_grid == color] = color_list[color_list.index(color) + 1]
        # If the count is 3 or less, change to black
        elif count <= 3:
            output_grid[input_grid == color] = Color.BLACK

    return output_grid