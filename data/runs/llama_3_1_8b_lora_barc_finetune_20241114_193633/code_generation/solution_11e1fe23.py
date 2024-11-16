from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mixing, blending, overlapping objects

# description:
# In the input, you will see two overlapping colored objects on a black background.
# To make the output, you should blend the colors of the two objects where they overlap, creating a new color in those areas.
# The blending should average the RGB values of the overlapping pixels (assuming colors are represented as integers).
# If there are no overlapping pixels, the output should be the original grid.

def blend_colors(color1, color2):
    # Simple averaging of color values assuming colors are represented as integers (0-9)
    if color1 == Color.BLACK:
        return color2
    if color2 == Color.BLACK:
        return color1
    return Color.RED  # Placeholder for blending logic (you can define your own blending logic here)

def transform(input_grid):
    # Plan:
    # 1. Create an output grid initialized to the input grid
    # 2. For each pixel in the grid, check if it overlaps with the two objects.
    # 3. If both colors are not black, blend the colors in that pixel.

    output_grid = np.copy(input_grid)
    n, m = input_grid.shape

    for x in range(n):
        for y in range(m):
            color1 = input_grid[x, y]
            if color1 == Color.BLACK:
                continue  # Skip black pixels

            # Check for the second object
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if 0 <= x + dx < n and 0 <= y + dy < m:
                        color2 = input_grid[x + dx, y + dy]
                        if color2!= Color.BLACK:
                            blended_color = blend_colors(color1, color2)
                            output_grid[x, y] = blended_color
                            break  # Stop once we have found a blend

    return output_grid