from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel manipulation, color transformation, boundary detection

# description:
# In the input grid, you will see a mixture of colored pixels and a black background. 
# The goal is to create an output grid where:
# 1. For each pixel that is surrounded by the same color on all sides (up, down, left, right), 
#    change its color to the color of the surrounding pixels. 
# 2. For each pixel that is isolated (not surrounded by the same color on all sides), 
#    change its color to the color of the majority of the surrounding pixels.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid for the output
    output_grid = np.copy(input_grid)
    
    # Get the shape of the grid
    rows, cols = input_grid.shape

    # Process each pixel in the grid
    for x in range(rows):
        for y in range(cols):
            # Check if the current pixel is not the background
            if input_grid[x, y]!= Color.BLACK:
                # Check if the pixel is surrounded by the same color on all sides
                is_surrounded = True
                if x > 0 and input_grid[x - 1, y] == input_grid[x, y]:
                    # Top
                    if y > 0 and input_grid[x, y - 1] == input_grid[x, y]:
                        # Left
                        pass
                    if y < cols - 1 and input_grid[x, y + 1] == input_grid[x, y]:
                        # Right
                        pass
                if x < rows - 1 and input_grid[x + 1, y] == input_grid[x, y]:
                    # Bottom
                    pass
                
                # If it has the same color on all sides, set it to that color
                if is_surrounded:
                    output_grid[x, y] = input_grid[x, y]
                else:
                    # Otherwise, check surrounding colors
                    surrounding_colors = []
                    if x > 0:
                        surrounding_colors.append(input_grid[x - 1, y])  # top
                    if x < rows - 1:
                        surrounding_colors.append(input_grid[x + 1, y])  # bottom
                    if y > 0:
                        surrounding_colors.append(input_grid[x, y - 1])  # left
                    if y < cols - 1:
                        surrounding_colors.append(input_grid[x, y + 1])  # right
                    # Count the occurrences of each color
                    color_counts = {}
                    for color in surrounding_colors:
                        if color!= Color.BLACK:
                            color_counts[color] = color_counts.get(color, 0) + 1
                    # If there are colors, find the majority color
                    if color_counts:
                        majority_color = max(color_counts, key=color_counts.get)
                        output_grid[x, y] = majority_color
                    else:
                        # No colors found, keep it black
                        output_grid[x, y] = Color.BLACK

    return output_grid