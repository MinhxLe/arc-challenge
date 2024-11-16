from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, color replacement

# description:
# In the input, you will see a grid with a pattern that has some symmetrical properties. 
# To create the output, replace all pixels that are not part of the symmetry with a specific color (e.g., blue) 
# while keeping the symmetrical parts unchanged.

def transform(input_grid):
    # Create an output grid initialized with the background color
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Detect translational symmetries in the grid
    symmetries = detect_translational_symmetry(input_grid, ignore_colors=[Color.BLACK])

    # Find all non-background pixels in the input grid
    for x, y in np.argwhere(input_grid!= Color.BLACK):
        # Check if the current pixel has a symmetrical counterpart
        for translation in orbit(input_grid, x, y, symmetries):
            if input_grid[translation]!= Color.BLACK:
                output_grid[x, y] = input_grid[x, y]  # Keep the color if it's symmetrical
                break

    # Replace non-symmetrical pixels with a specific color (e.g., blue)
    output_grid[output_grid == Color.BLACK] = Color.BLUE

    return output_grid