from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, pixel manipulation, spatial transformation

# description:
# In the input, you will see a grid filled with pixels of two colors: blue and teal. 
# The blue pixels represent the base color, while the teal pixels represent a highlight color. 
# To create the output grid, blend the blue and teal pixels together to produce a new color (e.g., teal + blue = teal) wherever they touch, 
# while keeping the rest of the pixels unchanged.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create an output grid that starts as a copy of the input grid
    output_grid = np.copy(input_grid)

    # Get the dimensions of the grid
    height, width = output_grid.shape

    # Iterate through each pixel in the grid
    for x in range(height):
        for y in range(width):
            # If the current pixel is blue, check its neighbors
            if output_grid[x, y] == Color.BLUE:
                # Check up, down, left, right
                if (x > 0 and output_grid[x - 1, y] == Color.PINK) or (x < height - 1 and output_grid[x + 1, y] == Color.PINK) or \
                   (y > 0 and output_grid[x, y - 1] == Color.PINK) or (y < width - 1 and output_grid[x, y + 1] == Color.PINK):
                    # If a neighbor is teal, blend colors to create teal
                    output_grid[x, y] = Color.PINK

    return output_grid