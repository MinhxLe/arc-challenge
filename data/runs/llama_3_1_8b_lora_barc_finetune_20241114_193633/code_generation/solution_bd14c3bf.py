from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel manipulation, symmetry detection

# description:
# In the input, you will see a grid filled with colored pixels, where some pixels are red and others are blue.
# The output grid should replace each red pixel with a corresponding blue pixel at the same position.
# If a red pixel has a blue pixel at a symmetric position (with respect to the center of the grid),
# that red pixel will be replaced by the blue pixel. The symmetry can be either horizontal, vertical, or both.

def transform(input_grid):
    # Get the dimensions of the input grid
    n, m = input_grid.shape
    
    # Create an output grid initialized to the input grid
    output_grid = np.copy(input_grid)

    # Calculate the center of the grid
    center_x, center_y = n // 2, m // 2

    # Check for horizontal symmetry
    for i in range(n):
        for j in range(m):
            if input_grid[i, j] == Color.RED:
                # Check for horizontal symmetry
                if (i < center_x and input_grid[n - 1 - i, j] == Color.BLUE) or (i >= center_x and input_grid[n - 1 - i, j] == Color.RED):
                    output_grid[i, j] = Color.BLUE

                # Check for vertical symmetry
                if (j < center_y and input_grid[i, m - 1 - j] == Color.BLUE) or (j >= center_y and input_grid[i, m - 1 - j] == Color.RED):
                    output_grid[i, j] = Color.BLUE

    return output_grid