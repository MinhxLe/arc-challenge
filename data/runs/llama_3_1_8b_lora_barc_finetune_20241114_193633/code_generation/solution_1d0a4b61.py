from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, color filling, mirroring

# description:
# In the input, you will see a grid with a symmetric pattern that has some missing parts. 
# To create the output, identify the missing parts and fill them in to restore the symmetry,
# using the colors from the existing pattern.

def transform(input_grid):
    # Detect mirror symmetries in the input grid
    symmetries = detect_mirror_symmetry(input_grid, ignore_colors=[Color.BLACK])
    output_grid = np.copy(input_grid)

    # For each pixel in the grid, check for symmetry and fill in missing colors
    for x, y in np.argwhere(input_grid!= Color.BLACK):
        for symmetry in symmetries:
            # Calculate the symmetric position
            symmetric_x, symmetric_y = symmetry.apply(x, y)
            # If the symmetric position is within bounds and is black, fill it with the current color
            if 0 <= symmetric_x < output_grid.shape[0] and 0 <= symmetric_y < output_grid.shape[1]:
                if output_grid[symmetric_x, symmetric_y] == Color.BLACK:
                    output_grid[symmetric_x, symmetric_y] = input_grid[x, y]

    return output_grid