from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color counting, symmetry detection

# description:
# In the input, you will see a grid with colored pixels. The grid contains one color that is the most dominant color. 
# To create the output, you need to create a new grid where:
# 1. All pixels of the dominant color are colored blue.
# 2. All pixels of the second most dominant color are colored teal.
# 3. All other colors are colored pink.
# 4. If a pixel is part of the dominant or second dominant color's boundary, it should also be colored with its respective color.

def transform(input_grid):
    # Count the occurrences of each color
    color_counts = {}
    for color in Color.NOT_BLACK:
        color_counts[color] = np.sum(input_grid == color)
    
    # Determine the dominant colors
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    dominant_color = sorted_colors[0][0]
    second_dominant_color = sorted_colors[1][0] if len(sorted_colors) > 1 else Color.BLACK

    # Create the output grid
    output_grid = np.full(input_grid.shape, Color.PINK)  # Start with pink color

    # Color the dominant color as blue
    output_grid[input_grid == dominant_color] = Color.BLUE

    # Color the second dominant color as teal
    output_grid[input_grid == second_dominant_color] = Color.TEAL

    # Color the boundary pixels of the dominant color blue
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            if input_grid[x, y] == dominant_color:
                # Check if the pixel is on the boundary
                if x == 0 or x == input_grid.shape[0] - 1 or y == 0 or y == input_grid.shape[1] - 1:
                    output_grid[x, y] = dominant_color

    # Color the boundary pixels of the second dominant color teal
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            if input_grid[x, y] == second_dominant_color:
                # Check if the pixel is on the boundary
                if x == 0 or x == input_grid.shape[0] - 1 or y == 0 or y == input_grid.shape[1] - 1:
                    output_grid[x, y] = second_dominant_color

    return output_grid