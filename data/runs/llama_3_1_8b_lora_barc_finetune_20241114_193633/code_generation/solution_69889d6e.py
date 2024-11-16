from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# connectivity, pathfinding, color propagation

# description:
# In the input, you will see a grid filled with a background color, and several colored pixels scattered throughout.
# The goal is to create an output grid where each colored pixel is expanded into a shape that represents its color.
# The shape is formed by connecting all neighboring pixels of the same color with lines.
# If two pixels of the same color are adjacent (including diagonally), they should be connected in the output grid.

def transform(input_grid):
    output_grid = np.copy(input_grid)
    visited = np.zeros_like(input_grid, dtype=bool)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

    for x, y in np.argwhere(input_grid!= Color.BLACK):
        if visited[x, y]:
            continue
        color = input_grid[x, y]
        queue = [(x, y)]
        visited[x, y] = True
        
        while queue:
            cx, cy = queue.pop(0)
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < output_grid.shape[0] and 0 <= ny < output_grid.shape[1]:
                    if (nx, ny) not in queue and output_grid[nx, ny] == Color.BLACK and not visited[nx, ny]:
                        queue.append((nx, ny))
                        visited[nx, ny] = True
                        output_grid[nx, ny] = color

    return output_grid