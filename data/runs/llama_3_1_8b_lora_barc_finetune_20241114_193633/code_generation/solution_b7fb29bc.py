from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, distance transformation

# description:
# In the input, you will see a grid with a colored object and a background color. 
# To create the output, color the pixels in the object based on their distance from the center of the object. 
# The closer pixels to the center will be colored with one color, and the farther pixels will be colored with another color.

def transform(input_grid):
    # Step 1: Detect the object in the input grid
    objects = detect_objects(input_grid, monochromatic=True, background=Color.BLACK, connectivity=4)

    # There should be exactly one object detected in the grid.
    assert len(objects) == 1
    object_sprite = objects[0]

    # Step 2: Find the bounding box of the object
    x, y, width, height = bounding_box(object_sprite, background=Color.BLACK)

    # Step 3: Create output grid, initialized to the background color
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Step 4: Compute the center of the object
    center_x = x + width // 2
    center_y = y + height // 2

    # Step 5: Calculate distance from the center and color accordingly
    for i in range(height):
        for j in range(width):
            if object_sprite[i, j]!= Color.BLACK:  # Only process colored pixels
                distance = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
                # Normalize distance to get a color based on distance
                if distance < 2:  # Closer to the center
                    output_grid[i, j] = Color.YELLOW
                else:  # Farther away
                    output_grid[i, j] = Color.RED

    return output_grid