from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# rotation, symmetry detection, color transformation

# description:
# In the input, you will see a grid with a colored shape that has rotational symmetry.
# To create the output, rotate the shape by 90 degrees clockwise and color the newly added pixels in a distinct color to indicate the rotation.

def transform(input_grid):
    # Find the rotational symmetry
    sym = detect_rotational_symmetry(input_grid, ignore_colors=[Color.BLACK])
    
    # Get the bounding box of the shape
    x, y, width, height = bounding_box(input_grid!= Color.BLACK)
    
    # Crop the shape from the input grid
    shape = crop(input_grid[y:y + height, x:x + width])
    
    # Create an output grid initialized to black
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Place the original shape in the output grid
    blit_sprite(output_grid, shape, x, y, background=Color.BLACK)
    
    # Rotate the shape by 90 degrees clockwise
    rotated_shape = np.rot90(shape, -1)

    # Color the newly added pixels with a distinct color (Color.YELLOW)
    new_x, new_y = y, x + width  # Position for the rotated shape
    blit_sprite(output_grid, rotated_shape, new_x, new_y, background=Color.BLACK)

    # Color the newly added pixels in the output grid
    output_grid[new_x:new_x + rotated_shape.shape[0], new_y:new_y + rotated_shape.shape[1]] = Color.YELLOW
    
    return output_grid