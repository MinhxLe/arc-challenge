from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel manipulation, expansion, color mixing

# description:
# In the input, you will see a grid with a single colored object surrounded by a blue background. 
# To create the output, expand the object outward by one pixel in all directions, mixing the colors of the 
# expanded pixels with the blue background. The mixing should produce a new color based on the average of the 
# original color and blue (Color.BLUE). The output should be a larger grid with the expanded object.

def transform(input_grid):
    # Create an output grid initialized to the background color (blue)
    output_grid = np.full(input_grid.shape, Color.BLUE)

    # Find the object in the input grid
    objects = find_connected_components(input_grid, background=Color.BLUE)
    
    # Assuming there's only one object in the input
    assert len(objects) == 1
    original_object = objects[0]

    # Get the color of the original object
    original_color = np.unique(original_object[original_object!= Color.BLUE])[0]

    # Get the bounding box of the original object
    x, y, width, height = bounding_box(original_object, background=Color.BLUE)

    # Expand the object outward by one pixel in all directions
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:
                continue  # Skip the center pixel

            # Calculate the coordinates of the new pixel
            new_x = x + dx
            new_y = y + dy

            # Check if the new coordinates are within bounds
            if 0 <= new_x < input_grid.shape[0] and 0 <= new_y < input_grid.shape[1]:
                # Mix the original color with the blue background
                if input_grid[new_x, new_y] == Color.BLUE:
                    # Create a new color by mixing the original color with blue
                    mixed_color = average_color(original_color, Color.BLUE)
                    output_grid[new_x, new_y] = mixed_color

    # Copy the original object to the output grid
    blit_sprite(output_grid, original_object, x=x, y=y, background=Color.BLUE)

    return output_grid

def average_color(color1, color2):
    # Simple function to mix two colors by averaging their indices
    color1_index = Color.ALL_COLORS.index(color1)
    color2_index = Color.ALL_COLORS.index(color2)
    # Calculate average as an index in the Color enum
    return Color.ALL_COLORS[(color1_index + color2_index) // 2]