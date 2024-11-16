from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color filling, symmetry, pattern completion

# description:
# In the input grid, you will see a symmetric pattern of colored pixels with some pixels missing.
# To create the output grid, fill in the missing pixels to complete the symmetry,
# ensuring that the colors are consistent with the existing colors in the grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Get the shape of the input grid
    n, m = input_grid.shape
    
    # Create an output grid initialized to the input grid
    output_grid = np.copy(input_grid)

    # Find the center of the grid
    center_x, center_y = n // 2, m // 2

    # Define the symmetry points for reflection
    for x in range(n):
        for y in range(m):
            if input_grid[x, y]!= Color.BLACK:  # Only fill in if it's not a background pixel
                # Calculate the symmetric positions
                sym_x = n - x - 1
                sym_y = m - y - 1

                # Fill in the symmetric positions if they are within bounds
                if 0 <= sym_x < n and 0 <= sym_y < m:
                    output_grid[sym_x, sym_y] = input_grid[x, y]

                # Also fill in the center point if it's not already filled
                if (x, y) == (center_x, center_y):
                    output_grid[center_x, center_y] = input_grid[x, y]

    return output_grid