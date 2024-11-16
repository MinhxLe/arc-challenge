from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# boundary detection, color filling

# description:
# In the input, you will see a grid with a black background and various colored pixels forming a shape. 
# To create the output, detect the boundary of the shape formed by the colored pixels and fill in the area inside the boundary with a new color (e.g., red). 
# The boundary should be defined by the pixels that are not black and are surrounded on all sides by black pixels.

def transform(input_grid):
    # Create an output grid initialized to black
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Get the coordinates of all colored pixels
    colored_pixels = np.argwhere(input_grid!= Color.BLACK)

    # Detect the boundary of the shape by checking connectivity
    for x, y in colored_pixels:
        # Check all 4-way connectivity
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if 0 <= x + dx < input_grid.shape[0] and 0 <= y + dy < input_grid.shape[1]:
                if input_grid[x + dx, y + dy] == Color.BLACK:
                    # If the neighboring pixel is black, we've found a boundary
                    output_grid[x, y] = Color.RED  # Fill with red inside the boundary
                    break  # Move to the next colored pixel

    return output_grid