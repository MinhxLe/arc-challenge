from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel manipulation, color shifting, boundary checking

# description:
# In the input, you will see a grid containing colored pixels and a black background.
# The goal is to transform the grid by shifting the colors of all pixels that are connected 
# to any red pixel in the grid. Each pixel will change its color to the next color in a predefined sequence.
# If the pixel is already the last color in the sequence, it wraps around to the first color.

def transform(input_grid):
    # Create a copy of the input grid for output
    output_grid = np.copy(input_grid)

    # Define the color sequence
    color_sequence = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GRAY,
                     Color.PINK, Color.ORANGE, Color.PURPLE, Color.BROWN, Color.BLACK]

    # Find all red pixels
    red_pixels = np.argwhere(input_grid == Color.RED)

    # Define a function to find the next color in the sequence
    def next_color(color):
        if color == Color.BLACK:
            return color_sequence[-1]  # Wrap around to the last color
        return color_sequence[(color_sequence.index(color) + 1) % len(color_sequence)]

    # Apply the transformation
    for red_x, red_y in red_pixels:
        # Use flood fill to propagate color changes
        flood_fill(output_grid, red_x, red_y, next_color(Color.RED), connectivity=4)

    return output_grid