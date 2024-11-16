from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel expansion, symmetry detection

# description:
# In the input, you will see a grid containing a colored object surrounded by a black background. 
# The object has rotational symmetry, but it is partially occluded by black pixels.
# To create the output, expand the visible portion of the object outward symmetrically until it reaches 
# the edge of the grid or encounters another color.

def transform(input_grid):
    # Plan:
    # 1. Detect the rotational symmetry of the object.
    # 2. Find the visible part of the object.
    # 3. Expand the object outward symmetrically until hitting the edge of the grid or another color.

    output_grid = np.full_like(input_grid, Color.BLACK)  # Start with a black background
    symmetries = detect_rotational_symmetry(input_grid, ignore_colors=[Color.BLACK])
    assert symmetries is not None, "No rotational symmetry found"

    # Find the colored object
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=False, connectivity=4)
    visible_object = objects[0]  # We expect only one object in the input

    # Find the bounding box of the visible part of the object
    x, y, width, height = bounding_box(visible_object, background=Color.BLACK)

    # Expand the object symmetrically
    for i in range(1, max(width, height) + 1):
        for angle in range(4):  # Only need to check 4 rotations
            rotated_object = np.rot90(visible_object, k=angle)
            # Check if the rotated object can be placed in the output grid
            for rx in range(max(0, x - i), min(output_grid.shape[0], x + width + i)):
                for ry in range(max(0, y - i), min(output_grid.shape[1], y + height + i)):
                    if (rx >= 0 and rx < output_grid.shape[0] and
                        ry >= 0 and ry < output_grid.shape[1] and
                        output_grid[rx, ry] == Color.BLACK):
                        # Place the rotated object in the output grid
                        blit_sprite(output_grid, rotated_object, x=rx - i, y=ry - i, background=Color.BLACK)

    return output_grid