from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, pixel expansion

# description:
# In the input, you will see a grid containing a colored shape (not black) surrounded by a black background.
# To make the output, expand the shape outward by one pixel in all directions, blending the new pixels with the background color.

def transform(input_grid):
    # Get the shape of the input grid
    n, m = input_grid.shape

    # Create an output grid initialized to the background color
    output_grid = np.full((n, m), Color.BLACK)

    # Find the connected components in the input grid
    components = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)

    # Assuming there's only one main shape
    if components:
        main_shape = components[0]
        # Get the bounding box of the shape
        x, y, w, h = bounding_box(main_shape, background=Color.BLACK)

        # Get the color of the main shape
        shape_color = np.unique(main_shape)[1]  # Skip the background color

        # Expand the shape outward by one pixel in all directions
        for i in range(n):
            for j in range(m):
                if (i >= 0 and i < n and j >= 0 and j < m and
                    (i > 0 and output_grid[i - 1, j] == Color.BLACK) and
                    (i < n - 1 and output_grid[i + 1, j] == Color.BLACK) and
                    (j > 0 and output_grid[i, j - 1] == Color.BLACK) and
                    (j < m - 1 and output_grid[i, j + 1] == Color.BLACK)):
                    # Check if the pixel is part of the shape
                    if (x <= i < x + w) and (y <= j < y + h) and (0 <= i < n) and (0 <= j < m):
                        output_grid[i, j] = shape_color

    return output_grid