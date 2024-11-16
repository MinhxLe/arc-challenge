from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color transformation, symmetry, pattern recognition

# description:
# In the input, you will see a grid containing a colored pattern that is symmetric about the vertical axis. 
# The output should transform the colors of the pixels on one side of the symmetry to their complementary colors 
# while keeping the other side unchanged. The complementary color mapping is defined as follows:
# red <-> brown, green <-> brown, blue <-> orange, yellow <-> pink, and black remains unchanged.

def transform(input_grid):
    # Define the color mapping for complementary colors
    color_mapping = {
        Color.RED: Color.BROWN,
        Color.GREEN: Color.BROWN,
        Color.BLUE: Color.ORANGE,
        Color.YELLOW: Color.PINK,
        Color.BLACK: Color.BLACK
    }

    # Get the shape of the input grid
    n, m = input_grid.shape

    # Create the output grid
    output_grid = np.copy(input_grid)

    # Check for vertical symmetry and apply color transformation
    for i in range(n):
        for j in range(m // 2):
            if input_grid[i, j]!= Color.BLACK:
                # Get the corresponding pixel on the opposite side
                opposite_j = m - 1 - j
                # If the opposite pixel is not black, change its color to its complementary color
                if input_grid[i, opposite_j]!= Color.BLACK:
                    output_grid[i, opposite_j] = color_mapping.get(input_grid[i, j], input_grid[i, j])

    return output_grid