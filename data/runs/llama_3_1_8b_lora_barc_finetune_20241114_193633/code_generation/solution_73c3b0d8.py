from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, expansion, central symmetry

# description:
# In the input, you will see a grid with a central colored object and a red pixel at the center.
# To create the output, expand the color of the central object outward in all directions, 
# until it reaches the red pixel. The expansion should maintain the color of the central object 
# and blend with any existing colors it encounters.

def transform(input_grid):
    # Plan:
    # 1. Find the central colored object
    # 2. Identify the position of the red pixel
    # 3. Expand the color of the central object outward until it reaches the red pixel

    # Step 1: Detect the central object (the only non-black component)
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=False)
    assert len(objects) == 1  # There should only be one central object
    central_object = objects[0]

    # Step 2: Get the color of the central object
    central_color = central_object[central_object!= Color.BLACK][0]

    # Step 3: Find the position of the red pixel
    red_pixel_objects = detect_objects(grid=input_grid, colors=[Color.RED], monochromatic=True, connectivity=4, allowed_dimensions=[(1, 1)])
    assert len(red_pixel_objects) == 1
    red_pixel_object = red_pixel_objects[0]
    
    # Get the position of the red pixel
    red_x, red_y = object_position(red_pixel_object, background=Color.BLACK)

    # Create the output grid, initialized to black
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Fill the output grid with the central object's color
    for x in range(central_object.shape[0]):
        for y in range(central_object.shape[1]):
            if central_object[x, y]!= Color.BLACK:
                output_grid[x, y] = central_color

    # Expand the color to the red pixel
    for x in range(output_grid.shape[0]):
        for y in range(output_grid.shape[1]):
            if output_grid[x, y] == Color.RED:
                # Fill in the surrounding pixels in a way that blends colors
                for dx in range(-1, 2):  # -1, 0, 1
                    for dy in range(-1, 2):  # -1, 0, 1
                        if dx == 0 and dy == 0:
                            continue  # skip the center pixel
                        if 0 <= x + dx < output_grid.shape[0] and 0 <= y + dy < output_grid.shape[1]:
                            if output_grid[x + dx, y + dy]!= Color.BLACK:
                                # Blend the colors: average the colors
                                output_grid[x + dx, y + dy] = blend_colors(output_grid[x + dx, y + dy], central_color)

    return output_grid

def blend_colors(color1, color2):
    if color1 == Color.BLACK:
        return color2
    if color2 == Color.BLACK:
        return color1
    # Simple average of two colors (for simplicity, assuming colors are integers)
    return Color.RED  # This is a placeholder, actual blending logic can be implemented