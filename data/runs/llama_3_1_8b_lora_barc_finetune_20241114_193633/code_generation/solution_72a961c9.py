from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, gradient creation

# description:
# In the input, you will see a series of colored pixels arranged in a line. 
# To create the output, generate a smooth gradient of colors between each pair of adjacent pixels, 
# filling the area between them with a color that transitions smoothly from one color to the other.

def transform(input_grid):
    # Create a copy of the input grid to store the output
    output_grid = np.copy(input_grid)

    # Find all unique colors in the input grid, excluding the background color
    unique_colors = np.unique(input_grid)
    unique_colors = unique_colors[unique_colors!= Color.BLACK]

    # Get the indices of non-background pixels
    indices = np.argwhere(input_grid!= Color.BLACK)

    # For each pair of adjacent pixels, generate a gradient
    for i in range(len(indices) - 1):
        x1, y1 = indices[i]
        x2, y2 = indices[i + 1]

        # Get the colors of the adjacent pixels
        color1 = input_grid[x1, y1]
        color2 = input_grid[x2, y2]

        # Create a gradient between color1 and color2
        for j in range(y1, y2 + 1):
            # Calculate the ratio for the gradient
            ratio = (j - y1) / (y2 - y1) if y2 > y1 else (j - y1) / (y1 - y1)
            blended_color = blend_colors(color1, color2, ratio)
            output_grid[x1, j] = blended_color

    return output_grid


def blend_colors(color1, color2, ratio):
    """
    Blends two colors based on the ratio.
    This function assumes colors are represented as strings.
    """
    # Define a mapping of colors to their RGB values
    color_map = {
        Color.RED: np.array([255, 0, 0]),
        Color.GREEN: np.array([0, 255, 0]),
        Color.BLUE: np.array([0, 0, 255]),
        Color.YELLOW: np.array([255, 255, 0]),
        Color.ORANGE: np.array([255, 165, 0]),
        Color.PINK: np.array([255, 192, 203]),
        Color.PURPLE: np.array([0, 128, 128]),
        Color.BROWN: np.array([128, 0, 0]),
        Color.GRAY: np.array([128, 128, 128]),
        Color.BLACK: np.array([0, 0, 0]),
    }

    # Convert colors to RGB
    rgb1 = color_map[color1]
    rgb2 = color_map[color2]

    # Blend the colors
    blended_rgb = (1 - ratio) * rgb1 + ratio * rgb2

    # Find the closest color from the color_map
    closest_color = min(color_map.keys(), key=lambda c: np.linalg.norm(color_map[c] - blended_rgb))
    return closest_color