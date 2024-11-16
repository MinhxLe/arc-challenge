from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# rotation, pattern matching, color correspondence

# description:
# In the input, you will see a grid containing a pattern of colored pixels. The pattern is made up of a mix of colors.
# To create the output grid, you should rotate the pattern 90 degrees clockwise. 
# The output grid should maintain the same dimensions as the input grid.

def transform(input_grid):
    # Get the shape of the input grid
    n, m = input_grid.shape

    # Create an output grid initialized to the background color
    output_grid = np.full((n, m), Color.BLACK)

    # Rotate the input grid 90 degrees clockwise
    for i in range(n):
        for j in range(m):
            output_grid[j, n - 1 - i] = input_grid[i, j]

    return output_grid