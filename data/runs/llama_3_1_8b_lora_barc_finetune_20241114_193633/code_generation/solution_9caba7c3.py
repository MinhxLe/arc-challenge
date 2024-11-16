from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, color propagation, boundaries

# description:
# In the input grid, you will see a black background with colored pixels scattered throughout. 
# To make the output, replace all pixels of color A with color B, and for each pixel of color B,
# spread its color to its immediate neighbors (up, down, left, right) if those neighbors are black.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    # Define the colors to be used
    color_a = Color.RED
    color_b = Color.GREEN

    # Find all positions of color_a and color_b
    positions_a = np.argwhere(input_grid == color_a)
    positions_b = np.argwhere(input_grid == color_b)

    # Replace color_a with color_b in the output grid
    output_grid[positions_a[:, 0], positions_a[:, 1]] = color_b

    # Spread color_b to immediate neighbors (up, down, left, right)
    for x, y in positions_b:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_x, neighbor_y = x + dx, y + dy
            if 0 <= neighbor_x < output_grid.shape[0] and 0 <= neighbor_y < output_grid.shape[1]:
                if output_grid[neighbor_x, neighbor_y] == Color.BLACK:
                    output_grid[neighbor_x, neighbor_y] = color_b

    return output_grid