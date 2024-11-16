from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# layering, color blending

# description:
# In the input, you will see a grid with several layers of colors, some of which overlap. 
# To create the output, you need to apply a blending rule that combines the colors of overlapping pixels according to a predefined blending function.
# The blending rule is as follows:
# 1. If two colors overlap, the resulting color will be determined by mixing them using a simple average of their RGB values.
# 2. The resulting color will be rounded to the nearest color in the Color enumeration.

def blend_colors(color1, color2):
    # Define a simple RGB blending function
    color_mapping = {
        Color.BLUE: np.array([0, 0, 0],
                            dtype=int),
        Color.RED: np.array([255, 0, 0],
                           dtype=int),
        Color.GREEN: np.array([0, 255, 0],
                             dtype=int),
        Color.YELLOW: np.array([255, 255, 0],
                              dtype=int),
        Color.GRAY: np.array([128, 128, 128],
                            dtype=int),
        Color.PINK: np.array([255, 192, 203],
                            dtype=int),
        Color.ORANGE: np.array([255, 165, 0],
                               dtype=int),
        Color.PURPLE: np.array([0, 128, 128],
                             dtype=int),
        Color.BROWN: np.array([128, 0, 0],
                              dtype=int),
        Color.BLACK: np.array([0, 0, 0],
                              dtype=int),
    }

    # Get the RGB values of the colors
    r1, g1, b1 = color_mapping[color1]
    r2, g2, b2 = color_mapping[color2]

    # Average the colors
    r_avg = (r1 + r2) // 2
    g_avg = (g1 + g2) // 2
    b_avg = (b1 + b2) // 2

    # Find the closest color
    closest_color = min(color_mapping.keys(), key=lambda c: np.linalg.norm(np.array(color_mapping[c]) - np.array([r_avg, g_avg, b_avg]), axis=0))

    return closest_color

def transform(input_grid):
    # Create an output grid initialized to black
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Get the height and width of the grid
    height, width = input_grid.shape

    # Iterate over each pixel in the input grid
    for x in range(height):
        for y in range(width):
            current_color = input_grid[x, y]
            if current_color!= Color.BLACK:
                # Check for overlapping pixels
                overlapping_colors = []
                for dx in range(-1, 2):  # Check adjacent pixels (up, down, left, right)
                    for dy in range(-1, 2):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < height and 0 <= ny < width:
                            neighbor_color = input_grid[nx, ny]
                            if neighbor_color!= Color.BLACK:
                                overlapping_colors.append(neighbor_color)

                # If there are overlapping colors, blend them
                if overlapping_colors:
                    output_color = blend_colors(*overlapping_colors)
                    output_grid[x, y] = output_color

    return output_grid