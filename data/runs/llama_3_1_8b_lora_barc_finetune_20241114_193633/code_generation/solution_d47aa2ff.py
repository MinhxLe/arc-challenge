from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color shifting, boundary detection, filling

# description:
# In the input, you will see a grid filled with colored pixels and a black boundary around it. 
# To make the output, shift the colors of the pixels towards the nearest edge of the grid while maintaining their relative positions to the boundary. 
# Pixels that are already at the boundary should remain unchanged.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)
    rows, cols = output_grid.shape

    # Get the color of the boundary
    boundary_color = Color.BLACK

    for x in range(rows):
        for y in range(cols):
            if output_grid[x, y] == boundary_color:
                # If the pixel is part of the boundary, skip it
                continue

            # Find the nearest boundary in the 4 cardinal directions
            nearest_distance = float('inf')
            nearest_x, nearest_y = -1, -1

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and output_grid[nx, ny] == boundary_color:
                    distance = abs(dx) + abs(dy)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_x, nearest_y = nx, ny

            # Shift the color towards the nearest boundary
            if nearest_x!= -1 and nearest_y!= -1:
                output_grid[x, y] = output_grid[nearest_x, nearest_y]

    return output_grid