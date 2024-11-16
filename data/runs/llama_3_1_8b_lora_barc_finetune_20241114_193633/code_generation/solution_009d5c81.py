from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color transformation, object separation, boundary detection

# description:
# In the input, you will see a grid with a complex shape formed by two colors (e.g., red and blue).
# To make the output, separate the shape into two distinct objects: one for each color, and change the color of each object to a new color.
# The red object should be transformed to green and the blue object to yellow.

def transform(input_grid):
    # Create the output grid, initially empty
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Detect all connected components in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK)

    # Change the color of each object based on its original color
    for obj in objects:
        color = np.unique(obj[obj!= Color.BLACK])[0]  # Get the color of the object
        # Create a mask for the current object
        mask = (obj == color)

        # Determine the bounding box of the object
        x, y, width, height = bounding_box(obj, background=Color.BLACK)

        # Create a new object with the transformed color
        if color == Color.RED:
            new_color = Color.GREEN
        elif color == Color.BLUE:
            new_color = Color.YELLOW
        else:
            continue  # Ignore other colors

        # Create a new object with the transformed color
        new_obj = np.full((height, width), new_color)

        # Place the new object in the output grid
        blit_sprite(output_grid, new_obj, x, y, background=Color.BLACK)

    return output_grid