from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, merging, pattern generation

# description:
# In the input, you will see two overlapping patterns, each with a distinct color. 
# To create the output, merge these two patterns by blending their colors where they overlap, 
# creating a new pattern that reflects the blending of the two colors. 
# The output grid should be filled with the merged colors in the overlapping areas.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid to store the output
    output_grid = np.copy(input_grid)

    # Get the connected components from the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=8, monochromatic=True)

    # Assume there are exactly two distinct patterns
    if len(objects)!= 2:
        raise ValueError("Expected exactly two distinct patterns in the input grid.")

    # Get the two patterns
    pattern1 = objects[0]
    pattern2 = objects[1]

    # Create a mask for the overlapping area
    overlap_mask = (pattern1!= Color.BLACK) & (pattern2!= Color.BLACK)

    # Blend colors in the overlapping area
    for x, y in np.argwhere(overlap_mask):
        color1 = pattern1[x, y]
        color2 = pattern2[x, y]

        # Simple blending logic: Average the two colors
        blended_color = blend_colors(color1, color2)
        output_grid[x, y] = blended_color

    return output_grid


def blend_colors(color1: int, color2: int) -> int:
    """
    A simple function to blend two colors. For simplicity, we can define a basic blending mechanism.
    In a real scenario, this could be more complex depending on the color representation.
    """
    # This is a simple average for blending purposes
    if color1 == Color.BLACK:
        return color2
    elif color2 == Color.BLACK:
        return color1
    else:
        # Here you can define more complex blending rules, but for now, we'll just return the first color
        return color1