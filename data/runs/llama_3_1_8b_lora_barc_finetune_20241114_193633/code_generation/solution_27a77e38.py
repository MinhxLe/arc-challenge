from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, color filling

# description:
# In the input, you will see a grid with a symmetric pattern of colored pixels and some missing parts represented by black pixels.
# To make the output, fill in the missing parts of the pattern symmetrically with the same color as the adjacent pixels.

def transform(input_grid):
    # Identify the color of the existing pixels
    existing_colors = np.unique(input_grid[input_grid!= Color.BLACK])
    
    # Create a copy of the input grid to fill in the missing parts
    output_grid = input_grid.copy()

    # Get the dimensions of the grid
    rows, cols = input_grid.shape

    # Fill in the missing parts based on symmetry
    for i in range(rows):
        for j in range(cols):
            if input_grid[i, j] == Color.BLACK:
                # Check the symmetric positions
                # Horizontal symmetry
                if i < rows // 2:
                    opposite_i = rows - 1 - i
                    if output_grid[opposite_i, j]!= Color.BLACK:
                        output_grid[i, j] = output_grid[opposite_i, j]
                # Vertical symmetry
                if j < cols // 2:
                    opposite_j = cols - 1 - j
                    if output_grid[i, opposite_j]!= Color.BLACK:
                        output_grid[i, j] = output_grid[i, opposite_j]

    return output_grid