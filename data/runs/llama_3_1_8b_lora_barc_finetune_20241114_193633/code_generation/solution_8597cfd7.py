from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object merging, color blending

# description:
# In the input, you will see two colored objects on a black background. 
# To create the output, merge the two objects into one by blending their colors together, 
# where overlapping pixels will take the average color of the two overlapping pixels.

def transform(input_grid):
    # Detect the two objects in the input grid
    objects = detect_objects(input_grid, monochromatic=False, background=Color.BLACK)

    # Initialize the output grid with the same background
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Blend the colors of the two objects
    for obj in objects:
        # Find the coordinates of the object
        obj_coords = np.argwhere(obj!= Color.BLACK)
        for x, y in obj_coords:
            # If the pixel is part of an object, we need to blend with the background or other colors
            if obj[x, y]!= Color.BLACK:
                # If the pixel is already colored in the output, blend the colors
                if output_grid[x, y]!= Color.BLACK:
                    output_grid[x, y] = blend_colors(output_grid[x, y], obj[x, y])
                else:
                    output_grid[x, y] = obj[x, y]

    return output_grid

def blend_colors(color1, color2):
    # Simple average color blending function
    if color1 == Color.BLACK:
        return color2
    if color2 == Color.BLACK:
        return color1

    # Assuming colors are represented as integers, we average them as follows:
    return (color1 + color2) // 2