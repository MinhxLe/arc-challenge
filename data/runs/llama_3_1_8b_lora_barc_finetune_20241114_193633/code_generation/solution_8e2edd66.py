from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# pattern replication, symmetry, rotation

# description:
# In the input, you will see a small patterned object in the center of the grid. 
# To create the output, replicate this pattern across the entire grid in all four quadrants,
# rotating it 90 degrees clockwise for each quadrant.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find the central pattern in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK)
    
    # Assume there is only one object in the center
    assert len(objects) == 1
    central_pattern = objects[0]

    # Get the bounding box of the central pattern
    x, y, w, h = bounding_box(central_pattern)

    # Create the output grid which is larger than the input grid
    output_grid_size = (input_grid.shape[0] * 2, input_grid.shape[1] * 2)
    output_grid = np.full(output_grid_size, Color.BLACK)

    # Crop the central pattern
    sprite = crop(central_pattern, background=Color.BLACK)

    # Get the center position of the original pattern
    center_x = (output_grid.shape[0] // 2) - (w // 2)
    center_y = (output_grid.shape[1] // 2) - (h // 2)

    # Place the original pattern in the center
    blit_sprite(output_grid, sprite, x=center_x, y=center_y, background=Color.BLACK)

    # Rotate and place the pattern in the other quadrants
    for i in range(1, 4):
        # Rotate the sprite 90 degrees clockwise
        rotated_sprite = np.rot90(sprite, k=-i)

        # Determine the position for the rotated sprite
        if i == 1:
            x_offset = center_x - (rotated_sprite.shape[0] // 2)
            y_offset = center_y - (rotated_sprite.shape[1] // 2)
        elif i == 2:
            x_offset = center_x + (rotated_sprite.shape[0] // 2)
            y_offset = center_y - (rotated_sprite.shape[1] // 2)
        elif i == 3:
            x_offset = center_x - (rotated_sprite.shape[0] // 2)
            y_offset = center_y + (rotated_sprite.shape[1] // 2)

        # Blit the rotated sprite onto the output grid
        blit_sprite(output_grid, rotated_sprite, x=x_offset, y=y_offset, background=Color.BLACK)

    return output_grid