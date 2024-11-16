from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern overlay, color blending

# description:
# In the input, you will see a grid with a pattern of colored pixels. 
# To make the output, overlay a transparent version of the pattern over a black background, 
# blending the colors where they overlap to create a new color based on the average of the overlapping colors.

def transform(input_grid):
    # Create an output grid initialized to black
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Extract the pattern from the input grid
    pattern = crop(input_grid, background=Color.BLACK)

    # Get the dimensions of the pattern
    pattern_height, pattern_width = pattern.shape

    # Iterate through each pixel in the pattern
    for i in range(pattern_height):
        for j in range(pattern_width):
            # Get the color of the current pixel in the pattern
            current_color = pattern[i, j]

            # Check if the current pixel is not black
            if current_color!= Color.BLACK:
                # Calculate the position to place the pixel in the output grid
                output_x, output_y = i, j
                # Blend the color with the existing color in the output grid
                if output_grid[output_x, output_y] == Color.BLACK:
                    output_grid[output_x, output_y] = current_color
                else:
                    # Blend the current color with the existing color
                    existing_color = output_grid[output_x, output_y]
                    blended_color = blend_colors(existing_color, current_color)
                    output_grid[output_x, output_y] = blended_color

    return output_grid

def blend_colors(color1, color2):
    """
    Blend two colors to create a new color. This function averages the colors based on their perceived brightness.
    """
    if color1 == Color.BLACK:
        return color2
    if color2 == Color.BLACK:
        return color1
    # Calculate the average color
    # Here we can use a simple average for blending
    return Color.RED if (color1 == Color.RED and color2 == Color.BLUE) else color1  # This is a simple example; you can modify this logic for better blending