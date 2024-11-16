from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# scaling, color blending

# description:
# In the input, you will see a small pattern of colored pixels. 
# To create the output, scale the pattern up by a factor of two and blend the colors of overlapping pixels
# by averaging their color values.

def transform(input_grid):
    # Get the dimensions of the input grid
    original_height, original_width = input_grid.shape
    
    # Scale the pattern up by a factor of 2
    scaled_height, scaled_width = original_height * 2, original_width * 2
    output_grid = np.full((scaled_height, scaled_width), Color.BLACK)

    # Fill the output grid by averaging colors for overlapping pixels
    for i in range(original_height):
        for j in range(original_width):
            # Check if the current pixel is not the background
            if input_grid[i, j]!= Color.BLACK:
                # Set the scaled positions in the output grid
                output_grid[2 * i, 2 * j] = input_grid[i, j]
                output_grid[2 * i + 1, 2 * j] = input_grid[i, j]
                output_grid[2 * i, 2 * j + 1] = input_grid[i, j]
                output_grid[2 * i + 1, 2 * j + 1] = input_grid[i, j]

    # Now we need to blend colors in overlapping pixels
    for i in range(scaled_height):
        for j in range(scaled_width):
            if output_grid[i, j]!= Color.BLACK:
                # If there's a color present, we check surrounding pixels
                neighbors = []
                if i > 0: neighbors.append(output_grid[i - 1, j])  # Up
                if i < scaled_height - 1: neighbors.append(output_grid[i + 1, j])  # Down
                if j > 0: neighbors.append(output_grid[i, j - 1])  # Left
                if j < scaled_width - 1: neighbors.append(output_grid[i, j + 1])  # Right

                # Filter out background colors
                neighbors = [color for color in neighbors if color!= Color.BLACK]

                # Average the colors
                if neighbors:
                    # Create a dictionary to count colors
                    color_count = {}
                    for color in neighbors:
                        if color not in color_count:
                            color_count[color] = 0
                        color_count[color] += 1
                    
                    # Find the most frequent color
                    avg_color = max(color_count, key=color_count.get)
                    output_grid[i, j] = avg_color

    return output_grid