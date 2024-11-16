from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# color blending, layering, pixel manipulation

# description:
# In the input, you will see a grid with two distinct layers of colors. The top layer is a pattern of colored pixels
# and the bottom layer is a solid color (black). To create the output grid, you should replace each colored pixel in the
# top layer with a new color based on the average of its neighboring pixels (up, down, left, right).
# If the neighboring pixels are the same color, it remains unchanged.

def transform(input_grid):
    # Plan:
    # 1. Create an output grid initialized to the background color
    # 2. Iterate over each pixel in the input grid
    # 3. For each colored pixel in the top layer, compute the average color of its neighbors
    # 4. Replace the colored pixel with the averaged color in the output grid

    output_grid = np.copy(input_grid)

    # Define the background color
    background = Color.BLACK

    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            # Check if the current pixel is not the background
            if input_grid[x, y]!= background:
                # Collect the colors of neighboring pixels
                neighbor_colors = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < input_grid.shape[0] and 0 <= ny < input_grid.shape[1]:
                        neighbor_colors.append(input_grid[nx, ny])
                
                # Calculate the average color
                avg_color = np.mean([Color.ALL_COLORS if color == background else color for color in neighbor_colors if color!= background])
                # Set the output pixel to the averaged color
                output_grid[x, y] = avg_color

    return output_grid