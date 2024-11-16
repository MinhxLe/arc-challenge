from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, symmetry, rotation

# description:
# In the input, you will see a grid with various colors forming a symmetric pattern.
# To create the output, you will need to rotate the colors of the grid around its center point.
# Each color should be mapped to a new color according to the following mapping:
# red -> green, blue -> yellow, green -> red, yellow -> blue, pink -> purple, purple -> brown, brown -> gray, gray -> pink.

def transform(input_grid):
    # Initialize output grid
    output_grid = input_grid.copy()

    # Constructing the color map
    color_map = {
        Color.RED: Color.GREEN,
        Color.BLUE: Color.YELLOW,
        Color.GREEN: Color.RED,
        Color.YELLOW: Color.BLUE,
        Color.PINK: Color.PURPLE,
        Color.PURPLE: Color.BROWN,
        Color.BROWN: Color.GRAY,
        Color.GRAY: Color.PINK
    }

    # Get the dimensions of the grid
    n, m = output_grid.shape

    # Calculate the center point
    center_x, center_y = n // 2, m // 2

    # Rotate and map colors
    for x in range(n):
        for y in range(m):
            if output_grid[x, y] in color_map:
                # Calculate the distance from the center
                distance_x = abs(center_x - x)
                distance_y = abs(center_y - y)
                
                # Rotate the position
                new_x = center_x - distance_x
                new_y = center_y - distance_y

                # Check if the new position is within bounds
                if 0 <= new_x < n and 0 <= new_y < m:
                    # Map the color
                    output_grid[new_x, new_y] = color_map[output_grid[x, y]]

    return output_grid