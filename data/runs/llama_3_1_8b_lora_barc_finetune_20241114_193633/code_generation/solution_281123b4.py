from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, averaging, adjacency

# description:
# In the input, you will see a grid with a pattern of colored pixels. 
# To make the output, you should create a new grid where each pixel is the average color of its neighboring pixels (up, down, left, right) from the input grid.
# If a neighboring pixel is black, it should not be considered in the averaging. 
# The output grid should be the same size as the input grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.zeros_like(input_grid)

    # Get the dimensions of the grid
    rows, cols = input_grid.shape

    for x in range(rows):
        for y in range(cols):
            # Collect neighboring colors, ignoring black
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and input_grid[nx, ny]!= Color.BLACK:
                    neighbors.append(input_grid[nx, ny])
            if neighbors:
                # Calculate the average color
                avg_color = np.mean([color for color in Color.ALL_COLORS if color in neighbors], axis=0).astype(int)
                output_grid[x, y] = avg_color[0]  # Assuming color is a single value, take the first one
            else:
                output_grid[x, y] = Color.BLACK  # Default to black if no neighbors

    return output_grid