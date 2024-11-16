from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, object transformation

# description:
# In the input, you will see a grid with colored objects on a black background. Each object has a specific color.
# To create the output grid, you should transform each object based on the following rules:
# - For every object of color A, change it to color B.
# - For every object of color C, change it to color D.
# - For every object of color E, change it to color F.
# The output grid should maintain the original position of each object in the input.

def transform(input_grid):
    # Create a copy of the input grid to serve as the output grid
    output_grid = np.copy(input_grid)
    
    # Define the color mapping
    color_mapping = {
        Color.RED: Color.GREEN,
        Color.BLUE: Color.YELLOW,
        Color.GREEN: Color.RED,
        Color.YELLOW: Color.BLUE,
        Color.PURPLE: Color.BROWN,
        Color.BROWN: Color.PURPLE,
        Color.PINK: Color.GRAY,
        Color.GRAY: Color.PINK,
        Color.ORANGE: Color.BLACK,
        Color.BLACK: Color.BLUE
    }

    # Iterate through the grid and apply the color mapping
    for x in range(output_grid.shape[0]):
        for y in range(output_grid.shape[1]):
            current_color = output_grid[x, y]
            if current_color in color_mapping:
                output_grid[x, y] = color_mapping[current_color]
    
    return output_grid