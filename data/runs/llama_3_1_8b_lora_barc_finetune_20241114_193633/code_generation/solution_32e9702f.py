from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# shape rotation, color inversion, symmetry

# description:
# In the input, you will see a colored shape on a black background.
# To make the output, rotate the shape 90 degrees clockwise and invert its colors, placing the new shape in the same position.

def transform(input_grid):
    # Find the bounding box of the shape
    x, y, width, height = bounding_box(input_grid)
    
    # Extract the shape from the grid
    shape = input_grid[x:x + width, y:y + height]
    
    # Rotate the shape 90 degrees clockwise
    rotated_shape = np.rot90(shape, k=-1)  # k=-1 for clockwise rotation
    
    # Invert the colors of the rotated shape
    inverted_shape = np.where(rotated_shape == Color.BLACK, Color.GRAY, Color.BLACK)  # Assume black is the background

    # Create an output grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Place the inverted shape in the same position
    blit_sprite(output_grid, inverted_shape, x=x, y=y, background=Color.BLACK)

    return output_grid