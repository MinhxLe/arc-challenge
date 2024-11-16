from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, pattern generation

# description:
# In the input, you will see a grid of colored pixels where some pixels are black and others are colored.
# To create the output, map each colored pixel to a new color based on a predefined mapping, 
# and then extend the color to fill the entire row and column of each colored pixel.

def transform(input_grid):
    # Initialize output grid
    output_grid = input_grid.copy()

    # Define the color mapping
    color_mapping = {
        Color.RED: Color.BLUE,
        Color.BLUE: Color.GREEN,
        Color.GREEN: Color.YELLOW,
        Color.YELLOW: Color.RED,
        Color.PURPLE: Color.BROWN,
        Color.BROWN: Color.PINK,
        Color.PINK: Color.GRAY,
        Color.GRAY: Color.ORANGE,
        Color.ORANGE: Color.BLACK,
        Color.BLACK: Color.BLACK  # Black remains unchanged
    }

    # Get the dimensions of the grid
    n, m = input_grid.shape

    # Iterate through the grid
    for x in range(n):
        for y in range(m):
            current_color = input_grid[x, y]
            if current_color in color_mapping:
                new_color = color_mapping[current_color]
                
                # Fill the entire row and column
                output_grid[x, :] = new_color  # Fill the entire row
                output_grid[:, y] = new_color  # Fill the entire column

    return output_grid