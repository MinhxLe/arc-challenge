from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# fractal generation, recursive patterns

# description:
# In the input, you will see a simple geometric pattern. 
# To create the output, recursively repeat the pattern in a fractal manner,
# each time scaling it down and rotating it slightly, until a specified number of iterations
# or the grid is filled.

def transform(input_grid):
    # Get the bounding box of the original pattern
    bounding_box_coords = bounding_box(input_grid)
    x, y, width, height = bounding_box_coords

    # Extract the original pattern
    pattern = input_grid[x:x + width, y:y + height]

    # Create the output grid, which will be larger than the original pattern
    output_size = 2 * max(width, height)
    output_grid = np.full((output_size, output_size), Color.BLACK)

    # Define the rotation angle increment for fractal generation
    rotation_increment = 45  # degrees
    iterations = 3  # number of iterations

    for _ in range(iterations):
        # Calculate the center position to place the pattern
        center_x = output_size // 2
        center_y = output_size // 2

        # Scale down the pattern
        scaled_pattern = scale_pattern(pattern, scale_factor=0.5)

        # Rotate the pattern
        rotated_pattern = np.rot90(scaled_pattern, k=_)

        # Calculate the position to place the rotated pattern
        x_offset = center_x - rotated_pattern.shape[0] // 2
        y_offset = center_y - rotated_pattern.shape[1] // 2

        # Blit the rotated pattern onto the output grid
        blit_sprite(output_grid, rotated_pattern, x=x_offset, y=y_offset, background=Color.BLACK)

        # Update the pattern for the next iteration
        pattern = rotated_pattern

    return output_grid