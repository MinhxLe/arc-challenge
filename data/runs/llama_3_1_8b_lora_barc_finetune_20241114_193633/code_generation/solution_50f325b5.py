from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# pixel manipulation, color mapping, object detection

# description:
# In the input, you will see a grid with several colored pixels scattered throughout. 
# To make the output, create a new grid where each pixel is replaced by its neighboring pixels' color:
# - If a pixel is surrounded by the same color pixels, it should be replaced by that color.
# - If a pixel has at least one neighboring pixel of a different color, it should be replaced with the color of the nearest different colored pixel.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)
    height, width = input_grid.shape

    # Iterate through each pixel in the grid
    for x in range(height):
        for y in range(width):
            # Check neighbors
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue  # skip the current pixel
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < height and 0 <= ny < width:
                        neighbors.append(input_grid[nx, ny])
            
            # Determine unique colors in neighbors
            unique_colors = set(neighbors)
            if Color.BLACK in unique_colors:
                unique_colors.remove(Color.BLACK)

            # If there are unique colors, find the nearest one
            if len(unique_colors) > 0:
                nearest_color = min(unique_colors, key=lambda c: (x, y, c))  # Using tuple comparison for sorting
                output_grid[x, y] = nearest_color
            else:
                output_grid[x, y] = input_grid[x, y]  # No neighbors of different colors

    return output_grid