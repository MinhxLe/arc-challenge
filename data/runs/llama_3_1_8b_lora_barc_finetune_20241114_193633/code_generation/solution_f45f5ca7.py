from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel shifting, wrapping, color transformation

# description:
# In the input, you will see a grid with colored pixels arranged in a specific pattern. The output grid should have the same dimensions as the input grid.
# The task is to shift the colored pixels to the right by a fixed number of positions. If a pixel moves out of bounds, it wraps around to the leftmost position in the same row.
# Additionally, if a pixel encounters another pixel of the same color during the shift, it should change to a new color.

def transform(input_grid):
    # Create an output grid of the same shape as the input grid
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Get the number of rows and columns
    rows, cols = input_grid.shape

    # Define the shift amount
    shift_amount = 2  # Fixed shift amount

    for x in range(rows):
        for y in range(cols):
            current_color = input_grid[x, y]
            if current_color!= Color.BLACK:  # Only shift non-background pixels
                new_y = (y + shift_amount) % cols  # Wrap around using modulo
                # If the new position is already occupied by the same color, change to a new color
                if output_grid[x, new_y] == current_color:
                    output_grid[x, new_y] = Color.RED  # Change to red as the new color
                else:
                    output_grid[x, new_y] = current_color  # Place the current color in the new position

    return output_grid