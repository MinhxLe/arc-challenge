from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# shape detection, rotation

# description:
# In the input, you will see a grid containing a shape made of colored pixels on a black background.
# To create the output, you should rotate the shape by 90 degrees clockwise and place it back onto the grid,
# ensuring it fits within the same bounding box.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find the bounding box of the shape in the input grid
    x, y, width, height = bounding_box(input_grid, background=Color.BLACK)
    
    # Extract the shape from the grid
    shape = crop(input_grid, background=Color.BLACK)

    # Create an empty output grid of the same size
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Rotate the shape 90 degrees clockwise
    rotated_shape = np.rot90(shape, k=-1)

    # Determine the new position to place the rotated shape in the output grid
    new_x = x + (width - rotated_shape.shape[1]) // 2
    new_y = y + (height - rotated_shape.shape[0]) // 2

    # Blit the rotated shape onto the output grid
    blit_sprite(output_grid, rotated_shape, x=new_x, y=new_y, background=Color.BLACK)

    return output_grid