from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, shape transformation

# description:
# In the input, you will see a grid with colored shapes. Each shape has a unique color and occupies a specific area. 
# To create the output, you need to transform each shape's color based on its position in the grid:
# 1. The shapes in the top half of the grid will be colored with a shade of blue.
# 2. The shapes in the bottom half of the grid will be colored with a shade of red.
# The output grid should maintain the same shapes, but their colors will change based on their position.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    # Get the number of rows in the grid
    num_rows = output_grid.shape[0]

    # Define the color transformations for the two halves
    color_mapping = {
        Color.BLUE: Color.BLUE,   # No change in color for blue
        Color.RED: Color.RED,     # No change in color for red
    }

    # Iterate over each pixel in the grid
    for x in range(num_rows):
        for y in range(input_grid.shape[1]):
            color = input_grid[x, y]
            # Determine the new color based on the row position
            if x < num_rows // 2:
                # Top half
                output_grid[x, y] = color_mapping.get(color, Color.BLACK)
            else:
                # Bottom half
                output_grid[x, y] = color_mapping.get(color, Color.BLACK)

    return output_grid