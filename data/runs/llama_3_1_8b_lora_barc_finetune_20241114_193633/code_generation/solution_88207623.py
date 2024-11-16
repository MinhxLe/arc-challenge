from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, filling, connected components

# description:
# In the input, you will see a grid with several colored objects on a black background.
# The goal is to transform the grid by filling each colored object with a new color based on its original color:
# - If the color is red, fill it with blue.
# - If the color is green, fill it with yellow.
# - If the color is blue, fill it with red.
# - If the color is yellow, fill it with green.
# - If the color is orange, fill it with black.
# - If the color is purple, fill it with orange.
# - If the color is gray, fill it with pink.
# - If the color is pink, fill it with gray.
# - If the color is brown, fill it with purple.
# If the color is black, it should remain unchanged.

def transform(input_grid):
    # Create a copy of the input grid to modify
    output_grid = np.copy(input_grid)

    # Define the color mapping
    color_mapping = {
        Color.RED: Color.BLUE,
        Color.GREEN: Color.YELLOW,
        Color.BLUE: Color.RED,
        Color.YELLOW: Color.GREEN,
        Color.ORANGE: Color.BLACK,
        Color.PURPLE: Color.ORANGE,
        Color.GRAY: Color.PINK,
        Color.PINK: Color.GRAY,
        Color.BROWN: Color.PURPLE,
        Color.BLACK: Color.BLACK,
    }

    # Detect all connected components in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)

    # Fill each object based on its color
    for obj in objects:
        # Get the color of the object
        color = obj[0, 0]  # Assuming the object is monochromatic
        new_color = color_mapping.get(color, color)  # Default to original color if not in mapping

        # Fill the object in the output grid with the new color
        for x, y in np.argwhere(obj!= Color.BLACK):
            output_grid[x, y] = new_color

    return output_grid