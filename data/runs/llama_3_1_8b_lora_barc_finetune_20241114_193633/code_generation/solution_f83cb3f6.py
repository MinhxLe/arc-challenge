from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel manipulation, connectivity, shape transformation

# description:
# In the input, you will see a grid filled with colored pixels on a black background. 
# The task is to transform the grid by creating a new grid where each colored pixel is replaced 
# by a corresponding pixel of a different color based on its distance from the nearest edge of the grid.
# The new color is determined by the Manhattan distance to the nearest edge (top, bottom, left, or right) 
# and is chosen from the color palette. The output grid should maintain the original shape of the input grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)
    height, width = output_grid.shape

    # Define the color palette
    color_palette = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GRAY, Color.PINK, Color.ORANGE, Color.PURPLE, Color.BROWN, Color.BLACK]

    # For each pixel in the grid, determine its distance to the nearest edge and set a new color
    for x in range(height):
        for y in range(width):
            if input_grid[x, y]!= Color.BLACK:
                # Calculate Manhattan distance to the nearest edge
                distances = {
                    'top': x,
                    'bottom': height - 1 - x,
                    'left': y,
                    'right': width - 1 - y
                }
                min_distance = min(distances, key=distances.get)

                # Set the new color based on the distance
                if min_distance == 'top':
                    output_grid[x, y] = Color.BLUE
                elif min_distance == 'bottom':
                    output_grid[x, y] = Color.RED
                elif min_distance == 'left':
                    output_grid[x, y] = Color.GREEN
                elif min_distance == 'right':
                    output_grid[x, y] = Color.YELLOW

    return output_grid