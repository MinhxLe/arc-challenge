from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# shape transformation, rotation, centering

# description:
# In the input, you will see a shape that is not centered in the grid. 
# To create the output, rotate the shape 90 degrees clockwise and center it in the grid, 
# filling the surrounding area with a background color.

def transform(input_grid):
    # Plan:
    # 1. Detect the shape in the input grid
    # 2. Rotate the shape 90 degrees clockwise
    # 3. Center the rotated shape in the output grid
    # 4. Fill the background with a specified background color

    # Detect the shape
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)
    shape = objects[0]  # Assuming there's only one shape

    # Crop the shape to isolate it
    cropped_shape = crop(shape, background=Color.BLACK)

    # Rotate the shape 90 degrees clockwise
    rotated_shape = np.rot90(cropped_shape, k=-1)  # k=-1 for clockwise rotation

    # Determine the size of the output grid
    output_height, output_width = rotated_shape.shape
    output_grid = np.full((output_height, output_width), Color.BLACK)  # Fill with background color

    # Center the rotated shape in the output grid
    center_x = (output_width - 1) // 2
    center_y = (output_height - 1) // 2
    blit_sprite(output_grid, rotated_shape, x=center_x, y=center_y, background=Color.BLACK)

    return output_grid