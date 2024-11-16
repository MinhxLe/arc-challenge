from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern recognition, color replacement, grid manipulation

# description:
# In the input, you will see a grid with a central pattern made of three colors (teal, blue, and yellow) 
# surrounded by a black background. The output should be a grid where the central pattern is rotated 
# 90 degrees clockwise and the colors are replaced with a new color scheme: 
# teal -> blue, blue -> teal, and yellow -> black.

def transform(input_grid):
    # Extract the central pattern by finding its bounding box
    objects = detect_objects(input_grid, colors=[Color.BLUE, Color.TEAL, Color.YELLOW], monochromatic=False, background=Color.BLACK)
    if not objects:
        return input_grid  # No objects found, return input as output

    central_pattern = crop(objects[0])  # Assume the first object is the central pattern

    # Rotate the pattern 90 degrees clockwise
    rotated_pattern = np.rot90(central_pattern, k=-1)

    # Create an output grid of the same size as the input
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Replace colors in the rotated pattern
    color_map = {
        Color.BLUE: Color.RED,
        Color.RED: Color.BLUE,
        Color.YELLOW: Color.GREEN
    }

    # Apply the color mapping
    for i in range(rotated_pattern.shape[0]):
        for j in range(rotated_pattern.shape[1]):
            original_color = rotated_pattern[i, j]
            if original_color in color_map:
                output_grid[i, j] = color_map[original_color]
            else:
                output_grid[i, j] = original_color

    return output_grid