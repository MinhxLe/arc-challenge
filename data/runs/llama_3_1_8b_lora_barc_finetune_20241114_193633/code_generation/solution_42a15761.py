from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, filling, pattern completion

# description:
# In the input, you will see a grid with a pattern that has been partially occluded by black pixels.
# The pattern has rotational symmetry around a central point. To create the output, identify the
# missing sections of the pattern and fill them in to complete the symmetrical design.

def transform(input_grid):
    # Identify the center of the grid
    center_x, center_y = input_grid.shape[0] // 2, input_grid.shape[1] // 2

    # Create a copy of the input grid to fill in the missing parts
    output_grid = np.copy(input_grid)

    # Check for rotational symmetry around the center
    for x, y in np.argwhere(input_grid!= Color.BLACK):
        # Calculate the corresponding symmetric positions
        sym_x = 2 * center_x - x
        sym_y = 2 * center_y - y
        
        # Fill in the missing pixels in the output grid
        if 0 <= sym_x < output_grid.shape[0] and 0 <= sym_y < output_grid.shape[1]:
            if output_grid[sym_x, sym_y] == Color.BLACK:
                output_grid[sym_x, sym_y] = input_grid[x, y]

    return output_grid