from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry, color blending

# description:
# In the input, you will see a grid with a red shape and a blue shape that is partially overlapping.
# To create the output, merge the two shapes by averaging the colors of overlapping pixels to create a new color (e.g., purple), while keeping the original colors outside the overlap.

def blend_colors(color1, color2):
    # Define a simple blending function for colors
    if color1 == Color.RED and color2 == Color.BLUE:
        return Color.YELLOW  # Example blend result for red and blue
    if color1 == Color.BLUE and color2 == Color.RED:
        return Color.YELLOW
    return color1  # Return the first color if no blending is needed

def transform(input_grid):
    # Create an output grid initialized to the background color
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Get the dimensions of the grid
    rows, cols = input_grid.shape

    # Iterate through the grid to merge the red and blue shapes
    for x in range(rows):
        for y in range(cols):
            color1 = input_grid[x, y]
            if color1 == Color.RED:
                # If the pixel is red, keep it in the output grid
                output_grid[x, y] = Color.RED
            elif color1 == Color.BLUE:
                # If the pixel is blue, keep it in the output grid
                output_grid[x, y] = Color.BLUE
            elif color1!= Color.BLACK:
                # If it's neither red nor blue, blend the color
                output_grid[x, y] = blend_colors(output_grid[x, y], Color.RED)
                output_grid[x, y] = blend_colors(output_grid[x, y], Color.BLUE)

    return output_grid