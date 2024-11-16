from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, pixel averaging

# description:
# In the input, you will see a grid filled with colored pixels, and there will be a separate grid that contains colored pixels representing the colors to blend. 
# To create the output grid, blend the colors from the input grid with the colors in the separate grid. 
# Each pixel in the output grid will be the average of the colors from the input grid and the corresponding pixel in the blending grid, 
# but only if the pixel in the blending grid is not black.

def transform(input_grid: np.ndarray, blend_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    # Get the shape of the input grid
    n, m = input_grid.shape

    # Iterate through the grid
    for x in range(n):
        for y in range(m):
            color = input_grid[x, y]
            blend_color = blend_grid[x, y]

            # If the blending color is not black, we blend the two colors
            if blend_color!= Color.BLACK:
                # Calculate the average color
                blended_color = blend_colors(color, blend_color)
                output_grid[x, y] = blended_color

    return output_grid

def blend_colors(color1, color2):
    # A simple function to blend two colors.
    # Here, we assume colors are represented as integers and blend by averaging their RGB values.
    # This is a simple example; you can modify it to blend colors based on their indices.
    return color1 if color1 == Color.BLACK else color2