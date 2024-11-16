from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, filling, mirroring

# description:
# In the input, you will see a grid with a colored shape on a black background.
# To create the output, check if the shape is symmetrical along the vertical axis.
# If it is symmetrical, fill the shape with a new color; if not, leave it unchanged.

def transform(input_grid):
    # Detect the shape by finding its bounding box
    x, y, width, height = bounding_box(input_grid, background=Color.BLACK)

    # Extract the shape
    shape = input_grid[x:x + width, y:y + height]

    # Check for vertical symmetry
    is_symmetric = np.array_equal(shape, shape[:, ::-1])

    # Create the output grid
    output_grid = np.copy(input_grid)

    if is_symmetric:
        # Fill the shape with a new color (for example, Color.GREEN)
        output_grid[x:x + width, y:y + height] = Color.GREEN

    return output_grid