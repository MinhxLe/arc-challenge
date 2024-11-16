from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, averaging, pixel manipulation

# description:
# In the input, you will see a grid filled with colored pixels, including some transparent pixels.
# To make the output, for each pixel that is not transparent, blend its color with the average color of its neighboring pixels (up, down, left, right).
# If a neighboring pixel is transparent, it should be ignored in the averaging. The resulting output should reflect the blended colors.

def transform(input_grid):
    # Get the shape of the input grid
    height, width = input_grid.shape
    
    # Create an output grid initialized to the background color
    output_grid = np.full((height, width), Color.BLACK)

    # Iterate through each pixel in the grid
    for x in range(height):
        for y in range(width):
            # Check if the current pixel is not the background
            if input_grid[x, y]!= Color.BLACK:
                # Initialize the color sum and count for averaging
                color_sum = np.zeros(3)  # Assuming RGB representation
                count = 0
                
                # Check neighboring pixels (up, down, left, right)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    # Check bounds
                    if 0 <= nx < height and 0 <= ny < width:
                        neighbor_color = input_grid[nx, ny]
                        if neighbor_color!= Color.BLACK:
                            color_sum += np.array([1, 0, 0])  # Assuming RGB representation
                            count += 1

                # If there are valid neighbors, compute the average color
                if count > 0:
                    avg_color = color_sum / count
                    output_grid[x, y] = Color.BLACK  # Reset to background before blending
                    # Blit the blended color (for simplicity, we assume Color is represented as RGB tuples)
                    output_grid[x, y] = Color.BLACK  # Placeholder for color blending

    return output_grid