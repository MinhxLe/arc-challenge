from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, color filling, pattern completion

# description:
# In the input, you will see a grid with a pattern that has been partially filled with random colors.
# To make the output, identify the symmetry of the pattern and fill in the missing parts to complete the symmetric design,
# coloring them with a specified color (e.g., green).

def transform(input_grid):
    # Identify the dimensions of the grid
    n, m = input_grid.shape
    
    # Create the output grid initialized to black
    output_grid = np.full((n, m), Color.BLACK)

    # Detect mirror symmetries in the input grid
    symmetries = detect_mirror_symmetry(input_grid, ignore_colors=[Color.BLACK])
    
    # Iterate through the grid to fill in the missing parts
    for x in range(n):
        for y in range(m):
            if input_grid[x, y]!= Color.BLACK:  # Only consider non-black pixels
                # Fill in the symmetric positions
                for sym in symmetries:
                    symmetric_x, symmetric_y = sym.apply(x, y)
                    if 0 <= symmetric_x < n and 0 <= symmetric_y < m:
                        output_grid[symmetric_x, symmetric_y] = input_grid[x, y]

    # Fill in the missing parts with green color
    output_grid[output_grid == Color.BLACK] = Color.GREEN

    return output_grid