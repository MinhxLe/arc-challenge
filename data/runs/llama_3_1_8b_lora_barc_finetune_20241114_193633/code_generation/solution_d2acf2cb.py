from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# symmetry detection, color inversion, pattern completion

# description:
# In the input, you will see a grid with a pattern that is partially obscured by black pixels.
# The pattern has rotational symmetry around its center. To create the output, identify the center of the
# pattern, remove the black pixels, and invert the colors of the visible pixels to create a new symmetrical pattern.

def transform(input_grid):
    # Step 1: Detect the rotational symmetry
    symmetries = detect_rotational_symmetry(input_grid, ignore_colors=[Color.BLACK])
    
    # Step 2: Crop the grid to find the visible part
    visible_part = crop(input_grid, background=Color.BLACK)

    # Step 3: Find the center of the visible part
    center_x, center_y = visible_part.shape[0] // 2, visible_part.shape[1] // 2

    # Step 4: Invert the colors of the visible part
    inverted_part = np.where(visible_part!= Color.BLACK, 
                            Color.NOT_BLACK - visible_part, 
                            Color.BLACK)  # Inverting the color

    # Step 5: Create the output grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Step 6: Apply the rotational symmetry to fill in the missing parts
    for x, y in np.argwhere(visible_part!= Color.BLACK):
        for i in range(4):  # 4 rotations
            rotated_x, rotated_y = symmetries.apply(x, y, iters=i)
            output_grid[rotated_x, rotated_y] = inverted_part[x, y]

    return output_grid