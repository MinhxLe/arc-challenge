from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# rotation, color mapping, symmetry

# description:
# In the input grid, you will see a colored object and a blue pixel at a specific location.
# To create the output grid, rotate the object 90 degrees clockwise around the blue pixel,
# and change its color to a new color based on its new orientation.

def transform(input_grid):
    # Step 1: Detect the blue pixel
    blue_pixel_objects = detect_objects(grid=input_grid, colors=[Color.BLUE], monochromatic=True, connectivity=4)
    assert len(blue_pixel_objects) == 1
    blue_pixel_object = blue_pixel_objects[0]

    # Step 2: Find the position of the blue pixel
    blue_x, blue_y = object_position(blue_pixel_object, background=Color.BLACK, anchor="upper left")

    # Step 3: Extract the colored object
    object_color = np.unique(input_grid[blit_object(input_grid, blue_pixel_object, background=Color.BLACK)])[0]
    object_pixels = np.argwhere(input_grid == object_color)

    # Step 4: Rotate the object 90 degrees clockwise
    new_grid = np.full(input_grid.shape, Color.BLACK)
    for x, y in object_pixels:
        # Calculate new position after rotation
        new_x = blue_x + (y - blue_y)
        new_y = blue_y - (x - blue_x)
        
        # Ensure new position is within bounds
        if 0 <= new_x < new_grid.shape[0] and 0 <= new_y < new_grid.shape[1]:
            new_grid[new_x, new_y] = object_color

    # Step 5: Change the color of the rotated object
    new_color = Color.RED if object_color == Color.BLUE else Color.BLUE  # Example color change logic

    # Step 6: Change the color of the rotated object
    output_grid = np.copy(new_grid)
    output_grid[new_grid == object_color] = new_color

    return output_grid