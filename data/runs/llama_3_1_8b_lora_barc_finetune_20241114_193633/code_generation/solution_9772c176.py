from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# shape transformation, rotation, symmetry, color mapping

# description:
# In the input, you will see a shape made of colored pixels. The shape will be rotated 90 degrees clockwise 
# to create a new output grid, and the color of the shape will be changed to match the color of the center pixel.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create an output grid of the same size, initialized to the background color
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Get the center pixel color
    center_x, center_y = input_grid.shape[0] // 2, input_grid.shape[1] // 2
    center_color = input_grid[center_x, center_y]

    # Rotate the grid 90 degrees clockwise
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            if input_grid[x, y]!= Color.BLACK:
                new_x = y
                new_y = input_grid.shape[0] - 1 - x
                output_grid[new_x, new_y] = center_color

    return output_grid