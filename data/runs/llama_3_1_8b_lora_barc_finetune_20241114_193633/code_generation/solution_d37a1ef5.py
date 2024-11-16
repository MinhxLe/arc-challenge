from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, color filling

# description:
# In the input, you will see a grid with a colored shape that is asymmetric. 
# To create the output, identify the axis of symmetry that would make the shape symmetric 
# and fill the non-symmetric parts with the color of the symmetric parts.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid for output
    output_grid = np.copy(input_grid)

    # Detect mirror symmetries
    mirrors = detect_mirror_symmetry(input_grid, ignore_colors=[Color.BLACK])

    # For each pixel in the grid, check for symmetry and fill the non-symmetric parts
    for x, y in np.argwhere(input_grid!= Color.BLACK):
        for mirror in mirrors:
            # Apply the mirror transformation
            symmetric_x, symmetric_y = mirror.apply(x, y)
            if (0 <= symmetric_x < output_grid.shape[0] and 0 <= symmetric_y < output_grid.shape[1]):
                if output_grid[symmetric_x, symmetric_y] == Color.BLACK:  # Only fill if it's a black pixel
                    output_grid[symmetric_x, symmetric_y] = input_grid[x, y]  # Fill with the color of the symmetric part

    return output_grid