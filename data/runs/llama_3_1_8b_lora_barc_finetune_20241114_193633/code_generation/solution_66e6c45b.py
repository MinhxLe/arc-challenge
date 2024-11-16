from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# radial symmetry, color matching

# description:
# In the input, you will see a pattern of colored pixels arranged in a circular fashion.
# To make the output, replicate the pattern in a way that maintains its radial symmetry, ensuring
# that the new arrangement is centered around the original center point of the input pattern.

def transform(input_grid):
    # Get the size of the input grid
    n, m = input_grid.shape

    # Calculate the center of the grid
    center_x, center_y = n // 2, m // 2

    # Extract the pattern around the center
    pattern = input_grid[center_x - 1:center_x + 1, center_y - 1:center_y + 1]

    # Create the output grid
    output_grid = np.full((n, m), Color.BLACK)

    # Calculate the radial symmetry positions
    for x in range(n):
        for y in range(m):
            if input_grid[x, y]!= Color.BLACK:
                # Calculate the distance from the center
                dx = x - center_x
                dy = y - center_y
                # Determine the symmetric positions
                sym_positions = [
                    (center_x + dx, center_y + dy),  # original position
                    (center_x - dx, center_y + dy),  # opposite
                    (center_x + dx, center_y - dy),  # right
                    (center_x - dx, center_y - dy)   # left
                ]
                # Blit the pattern in the symmetric positions
                for sx, sy in sym_positions:
                    if 0 <= sx < n and 0 <= sy < m:
                        blit_sprite(output_grid, pattern, x=sx - 1, y=sy - 1, background=Color.BLACK)

    return output_grid