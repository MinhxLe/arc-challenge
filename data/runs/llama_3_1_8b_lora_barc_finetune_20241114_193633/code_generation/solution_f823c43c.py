from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, gradient creation

# description:
# In the input, you will see a grid filled with two colors in a checkerboard pattern. 
# To create the output, blend the two colors in a gradient manner across the entire grid, 
# resulting in a smooth transition between the two colors.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Get the unique colors in the grid
    unique_colors = np.unique(input_grid)
    
    # Check if there are exactly two colors
    if len(unique_colors)!= 2:
        raise ValueError("Input grid must contain exactly two colors.")

    # Get the two colors
    color1, color2 = unique_colors[0], unique_colors[1]
    
    # Create an output grid initialized to black
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Get the dimensions of the grid
    height, width = input_grid.shape
    
    # Blend colors based on their positions
    for y in range(height):
        for x in range(width):
            if input_grid[y, x] == color1:
                output_grid[y, x] = color1
            elif input_grid[y, x] == color2:
                output_grid[y, x] = color2
            else:
                # Blend colors for the black pixels
                # Calculate the ratio based on the position
                ratio = (x + y) / (width + height)
                # Interpolate between the two colors
                blended_color = blend_colors(color1, color2, ratio)
                output_grid[y, x] = blended_color

    return output_grid


def blend_colors(color1, color2, ratio):
    # Simple linear interpolation between two colors
    return color1 if ratio < 0.5 else color2