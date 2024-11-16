from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, radial gradient

# description:
# In the input, you will see a grid filled with various colors forming a pattern. 
# To create the output, you will map each color in the input grid to a new color based on a radial gradient.
# The output grid should maintain the same structure but with the colors transformed according to the gradient.

def transform(input_grid):
    # Get the dimensions of the input grid
    height, width = input_grid.shape

    # Initialize the output grid
    output_grid = np.full((height, width), Color.BLACK)

    # Define a color gradient mapping function
    def color_gradient(color):
        # Assuming a basic gradient mapping where each color in the input maps to a new color based on its RGB values
        # Here we will use a simple mapping for demonstration purposes
        if color == Color.RED:
            return Color.BLUE
        elif color == Color.GREEN:
            return Color.YELLOW
        elif color == Color.BLUE:
            return Color.RED
        elif color == Color.YELLOW:
            return Color.ORANGE
        elif color == Color.ORANGE:
            return Color.GREEN
        elif color == Color.GRAY:
            return Color.PINK
        elif color == Color.PINK:
            return Color.PURPLE
        elif color == Color.PURPLE:
            return Color.BROWN
        elif color == Color.BROWN:
            return Color.GRAY
        elif color == Color.BLACK:
            return Color.BLACK
        else:
            return Color.BLACK  # Fallback

    # Apply the color gradient transformation
    for x in range(height):
        for y in range(width):
            output_grid[x, y] = color_gradient(input_grid[x, y])

    return output_grid