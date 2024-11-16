from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, object translation

# description:
# In the input, you will see a grid filled with colored objects on a black background. 
# The objects are represented as distinct colors and can overlap. 
# To create the output grid, you need to translate each object to the nearest edge of the grid while maintaining their original colors. 
# If two objects are adjacent to each other, they should not overlap after translation.

def transform(input_grid):
    # Create an output grid initialized to black
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Detect all objects in the input grid
    objects = detect_objects(input_grid, monochromatic=False, background=Color.BLACK)

    # For each object, find its color and its bounding box
    for obj in objects:
        # Get the bounding box of the current object
        x, y, w, h = bounding_box(obj, background=Color.BLACK)
        color = obj[x, y]

        # Translate the object to the nearest edge of the grid without overlapping
        while True:
            # Find a free position for the object (top-left corner)
            for dx in range(-1, w + 1):
                for dy in range(-1, h + 1):
                    new_x, new_y = x + dx, y + dy
                    if new_x < 0 or new_x >= output_grid.shape[0] or new_y < 0 or new_y >= output_grid.shape[1]:
                        continue
                    if output_grid[new_x, new_y] == Color.BLACK:
                        # Place the object in the new position
                        blit_sprite(output_grid, obj, new_x, new_y, background=Color.BLACK)
                        break
                    else:
                        continue
                else:
                    # If no free position found, reset and try again
                    continue
                break

    return output_grid