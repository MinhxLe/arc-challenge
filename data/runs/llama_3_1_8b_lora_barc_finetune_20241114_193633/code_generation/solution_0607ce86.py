from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, pattern recognition, adjacency

# description:
# In the input, you will see a grid containing various colored shapes. 
# The output grid should replace each color in the input with a new color based on the following mapping:
# - If a color is adjacent to a green pixel, it turns into blue.
# - If a color is adjacent to a red pixel, it turns into yellow.
# - If a color is adjacent to a blue pixel, it turns into green.
# The mapping applies to all connected components of the same color that are adjacent to the specified colors.

def transform(input_grid):
    # Initialize output grid
    output_grid = np.copy(input_grid)

    # Get the dimensions of the grid
    rows, cols = input_grid.shape

    # Create a mapping for colors based on adjacency
    for x in range(rows):
        for y in range(cols):
            color = input_grid[x, y]
            if color == Color.GREEN:
                # Change adjacent colors to blue
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if 0 <= x + dx < rows and 0 <= y + dy < cols:
                        output_grid[x + dx, y + dy] = Color.BLUE

            elif color == Color.RED:
                # Change adjacent colors to yellow
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if 0 <= x + dx < rows and 0 <= y + dy < cols:
                        output_grid[x + dx, y + dy] = Color.YELLOW

            elif color == Color.BLUE:
                # Change adjacent colors to green
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if 0 <= x + dx < rows and 0 <= y + dy < cols:
                        output_grid[x + dx, y + dy] = Color.GREEN

    return output_grid