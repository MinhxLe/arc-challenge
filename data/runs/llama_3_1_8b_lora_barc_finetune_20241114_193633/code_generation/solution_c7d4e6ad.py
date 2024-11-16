from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object detection, color mapping, symmetry

# description:
# In the input, you will see several colored objects scattered on a grid. 
# Each object has a specific color and occupies a contiguous area. 
# To make the output, you need to identify all objects and create a new grid where 
# each object's color is replaced by the color of the object that is mirrored about the center of the grid.
# The output grid should maintain the original structure of the objects, but with the colors changed according to the mirroring rule.

def transform(input_grid):
    # Plan:
    # 1. Find all connected components (objects) in the input grid.
    # 2. For each object, find its mirror position in the grid.
    # 3. Replace the color of the object with the color of its mirrored counterpart.

    # Step 1: Detect objects in the input grid
    objects = detect_objects(input_grid, monochromatic=False, connectivity=4, background=Color.BLACK)

    # Create a copy of the input grid for output
    output_grid = np.copy(input_grid)

    # Step 2: Mirror each object and replace its color
    for obj in objects:
        # Crop the object to get its sprite
        sprite = crop(obj, background=Color.BLACK)
        # Get the bounding box of the object
        x, y, w, h = bounding_box(sprite, background=Color.BLACK)

        # Calculate the mirrored position
        mirror_x = x + (x + w - 1) - (x - w // 2)  # Mirror about the center
        mirror_y = y + (y + h - 1) - (y - h // 2)  # Mirror about the center

        # Get the color of the mirrored object
        mirrored_color = input_grid[mirror_x, mirror_y]

        # Replace the color of the original object in the output grid
        output_grid[obj!= Color.BLACK] = mirrored_color

    return output_grid