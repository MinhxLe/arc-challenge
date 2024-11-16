from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, boundary detection

# description:
# In the input, you will see a grid with a black background and colored pixels. 
# The goal is to create an output grid where each pixel that is not black is mapped to the nearest colored pixel 
# of the same color in the grid. If there are multiple nearest colored pixels, the first one encountered in a 
# depth-first search (DFS) order will be chosen. If no colored pixel is found, it should remain black.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    # Get the dimensions of the grid
    rows, cols = input_grid.shape

    # Iterate through each pixel in the grid
    for x in range(rows):
        for y in range(cols):
            if input_grid[x, y] == Color.BLACK:
                # Initialize a queue for DFS
                queue = [(x, y)]
                visited = set(queue)

                # Perform DFS to find the nearest colored pixel
                while queue:
                    cx, cy = queue.pop(0)
                    # Check if the current position is within bounds and is not background
                    if (cx < 0 or cx >= rows or cy < 0 or cy >= cols or 
                        input_grid[cx, cy] == Color.BLACK or (cx, cy) in visited):
                        continue
                    
                    # Check if we found a colored pixel
                    if input_grid[cx, cy]!= Color.BLACK:
                        output_grid[x, y] = input_grid[cx, cy]
                        break

                    # Mark the current pixel as visited
                    visited.add((cx, cy)

                    # Check all 4 neighbors
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                            queue.append((nx, ny))

    return output_grid