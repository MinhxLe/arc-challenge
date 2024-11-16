from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object separation, color mapping, connectivity

# description:
# In the input, you will see a grid containing several colored objects, some of which may overlap. 
# Each object can be of any color and can be connected either horizontally or vertically. 
# To create the output, separate each object into its own region and color each region based on its original color, 
# using a mapping where the first object becomes blue, the second becomes red, the third becomes green, and so on, 
# cycling through the colors until all objects are colored.

def transform(input_grid):
    # Find all connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True, connectivity=4)

    # Initialize the output grid with the same shape as input
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Color mapping for the objects
    color_mapping = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GRAY, Color.PINK, Color.ORANGE, Color.PURPLE, Color.BROWN, Color.BLACK]
    color_index = 0

    # Iterate through each object found
    for obj in objects:
        # Get the color for the current object
        color = color_mapping[color_index]
        color_index = (color_index + 1) % len(color_mapping)

        # Crop the object to get its bounding box
        cropped_obj = crop(obj, background=Color.BLACK)

        # Get the bounding box of the object
        x, y, w, h = bounding_box(cropped_obj, background=Color.BLACK)

        # Create a mask of the object
        mask = np.full_like(cropped_obj, Color.BLACK)
        mask[cropped_obj!= Color.BLACK] = color

        # Find the position to place the colored object in the output grid
        output_x, output_y = y, x  # x and y are swapped because we want to place it in the original grid position

        # Place the colored object in the output grid
        blit_sprite(output_grid, mask, output_x, output_y, background=Color.BLACK)

        # Update the color index for the next object
        color_index = (color_index + 1) % len(color_mapping)

    return output_grid