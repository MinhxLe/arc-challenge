from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# rotation, color mapping

# description:
# In the input, you will see a grid with colored shapes, some of which are occluded by black pixels.
# The output should be a grid where the occluded shapes are revealed by rotating the visible shapes around their centers,
# while maintaining their color and relative positions.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid for output
    output_grid = np.copy(input_grid)

    # Detect all the colored objects in the grid
    objects = detect_objects(grid=input_grid, background=Color.BLACK, monochromatic=False, connectivity=8)

    for obj in objects:
        # Get the bounding box of the object
        x, y, w, h = bounding_box(obj)
        cropped_object = crop(obj, background=Color.BLACK)

        # Find the center of the object
        center_x, center_y = x + w // 2, y + h // 2

        # Rotate the object 90 degrees clockwise
        rotated_object = np.rot90(cropped_object, k=-1)  # k=-1 for clockwise rotation

        # Calculate the new position to place the rotated object
        new_x, new_y = center_x, center_y  # Place it back at the center

        # Clear the old position
        output_grid[x:x + w, y:y + h] = Color.BLACK

        # Blit the rotated object onto the output grid
        blit_object(output_grid, rotated_object, background=Color.BLACK)

    return output_grid