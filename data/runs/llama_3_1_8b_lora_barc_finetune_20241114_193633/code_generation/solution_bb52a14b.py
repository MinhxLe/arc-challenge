from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object merging, color transformation

# description:
# In the input, you will see a grid with a black background and a colored object that can overlap with other colored objects.
# To create the output, merge the colors of overlapping pixels by averaging their RGB values, creating a new color that represents the overlap.
# The output grid should reflect the merged colors where overlaps occur.

def transform(input_grid):
    # Create an output grid initialized to black
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Get the connected components of the input grid, treating black as background
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4)

    # Iterate over each object and merge colors in overlapping areas
    for obj in objects:
        # Get the color of the object
        obj_color = obj[0, 0]  # Assume the object is monochromatic
        # Find the coordinates of the object
        obj_coords = np.argwhere(obj!= Color.BLACK)

        # For each pixel in the object, if it overlaps with another object's pixel, merge colors
        for x, y in obj_coords:
            if output_grid[x, y]!= Color.BLACK:
                # Average the color with the existing color in the output grid
                output_grid[x, y] = average_colors(output_grid[x, y], obj_color)
            else:
                # If it's the first pixel, just set it to the object's color
                output_grid[x, y] = obj_color

    return output_grid

def average_colors(color1, color2):
    # Simple average function for colors represented as integers
    return (color1 + color2) // 2