from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# rotation, color mapping, pixel manipulation

# description:
# In the input, you will see a grid with colored pixels forming a shape. The shape consists of a specific color (e.g., teal).
# To create the output, rotate the shape 90 degrees clockwise and change its color according to the following mapping:
# teal -> blue, red -> green, green -> yellow, blue -> red, yellow -> green, orange -> purple, pink -> gray, purple -> brown, gray -> pink, brown -> orange.

def transform(input_grid):
    # Initialize output grid
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Find connected components (the teal shape)
    teal_objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)
    
    if not teal_objects:
        return output_grid  # No teal shape found, return empty grid

    # Assuming the first object found is the teal shape
    teal_shape = teal_objects[0]
    
    # Get the bounding box of the teal shape
    x, y, width, height = bounding_box(teal_shape, background=Color.BLACK)
    
    # Crop the teal shape from the grid
    shape_sprite = crop(teal_shape, background=Color.BLACK)

    # Rotate the shape 90 degrees clockwise
    rotated_shape = np.rot90(shape_sprite, k=-1)

    # Construct color mapping
    color_map = {
        Color.BLUE: Color.RED,
        Color.RED: Color.GREEN,
        Color.GREEN: Color.YELLOW,
        Color.YELLOW: Color.GREEN,
        Color.RED: Color.BLUE,
        Color.BLUE: Color.RED,
        Color.ORANGE: Color.PURPLE,
        Color.PURPLE: Color.BROWN,
        Color.BROWN: Color.PINK,
        Color.PINK: Color.ORANGE,
    }

    # Apply the color mapping
    for i in range(rotated_shape.shape[0]):
        for j in range(rotated_shape.shape[1]):
            if rotated_shape[i, j] in color_map:
                output_grid[y + i, x + j] = color_map[rotated_shape[i, j]]
            else:
                output_grid[y + i, x + j] = Color.BLACK  # Set to black for unmapped colors

    return output_grid