from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry, reflection, mirroring

# description:
# In the input, you will see a colored object on a grid, surrounded by a black background.
# To create the output, reflect the object across the center of the grid, filling in the mirrored pixels with the same color as the original object.

def transform(input_grid):
    # Create a copy of the input grid to manipulate
    output_grid = input_grid.copy()

    # Get the dimensions of the grid
    rows, cols = input_grid.shape

    # Calculate the center coordinates
    center_x, center_y = rows // 2, cols // 2

    # Get the color of the original object
    object_color = input_grid[input_grid!= Color.BLACK][0]

    # Reflect the object across the center
    for x in range(rows):
        for y in range(cols):
            if input_grid[x, y] == object_color:
                reflected_x = center_x - (x - center_x)
                reflected_y = center_y - (y - center_y)

                # Ensure the reflected coordinates are within bounds
                if 0 <= reflected_x < rows and 0 <= reflected_y < cols:
                    output_grid[reflected_x, reflected_y] = object_color

    return output_grid