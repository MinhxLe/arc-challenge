from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# rotation, color mapping, grid transformation

# description:
# In the input, you will see a grid containing a colored shape and a gray square at the top left corner. 
# The goal is to rotate the shape 90 degrees clockwise around the center of the gray square, 
# while maintaining the shape's colors. The output should be a grid showing the rotated shape.

def transform(input_grid):
    # Identify the position of the gray square
    gray_square_position = np.argwhere(input_grid == Color.GRAY)
    if gray_square_position.size == 0:
        return np.copy(input_grid)  # Return the original grid if gray square is not found

    # Get the coordinates of the gray square
    gray_x, gray_y = gray_square_position[0]

    # Extract the shape by cropping around the gray square
    shape = crop(input_grid, background=Color.BLACK)

    # Rotate the shape 90 degrees clockwise
    rotated_shape = np.rot90(shape, k=-1)

    # Determine the size of the output grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Place the rotated shape centered around the gray square
    x_offset = gray_x - (rotated_shape.shape[0] // 2)
    y_offset = gray_y - (rotated_shape.shape[1] // 2)
    output_grid[x_offset:x_offset + rotated_shape.shape[0], y_offset:y_offset + rotated_shape.shape[1]] = rotated_shape

    return output_grid