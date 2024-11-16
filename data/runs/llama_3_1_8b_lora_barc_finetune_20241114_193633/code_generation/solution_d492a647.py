from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# pixel manipulation, color mapping, connectivity

# description:
# In the input, you will see a grid filled with colored pixels. There will be a specific color (e.g., blue) that represents the "source" color.
# To make the output, for each pixel of the source color, replace it with a new color (e.g., green) and also change its immediate neighbors (up, down, left, right) to a different color (e.g., blue).
# If a neighbor is already the source color or another color, it should not be changed.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    # Get the source color and the target colors
    source_color = Color.BLUE
    target_color = Color.GREEN
    neighbor_color = Color.BLUE

    # Iterate through the grid
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            if input_grid[x, y] == source_color:
                # Change the source pixel to the target color
                output_grid[x, y] = target_color

                # Change the immediate neighbors to the neighbor color
                if x > 0:  # Up
                    if input_grid[x - 1, y]!= Color.BLACK and input_grid[x - 1, y]!= neighbor_color:
                        output_grid[x - 1, y] = neighbor_color
                if x < input_grid.shape[0] - 1:  # Down
                    if input_grid[x + 1, y]!= Color.BLACK and input_grid[x + 1, y]!= neighbor_color:
                        output_grid[x + 1, y] = neighbor_color
                if y > 0:  # Left
                    if input_grid[x, y - 1]!= Color.BLACK and input_grid[x, y - 1]!= neighbor_color:
                        output_grid[x, y - 1] = neighbor_color
                if y < input_grid.shape[1] - 1:  # Right
                    if input_grid[x, y + 1]!= Color.BLACK and input_grid[x, y + 1]!= neighbor_color:
                        output_grid[x, y + 1] = neighbor_color

    return output_grid