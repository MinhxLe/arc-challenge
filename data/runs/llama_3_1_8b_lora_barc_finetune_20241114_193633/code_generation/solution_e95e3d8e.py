from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, color restoration

# description:
# In the input, you will see a grid with a symmetric object that has been occluded by a black rectangle. 
# To create the output, detect the symmetry and fill in the missing parts of the object using the colors of the visible parts.

def transform(input_grid):
    # Plan:
    # 1. Detect the black rectangle (occlusion)
    # 2. Find the rotational symmetry of the visible part of the object
    # 3. Fill in the missing parts of the object using the detected symmetry

    # Step 1: Detect the black rectangle (occlusion)
    black_rectangle_mask = (input_grid == Color.BLACK)
    
    # Step 2: Find the rotational symmetry
    symmetries = detect_rotational_symmetry(input_grid, ignore_colors=[Color.BLACK])
    
    # Step 3: Fill in the missing parts
    for x, y in np.argwhere(black_rectangle_mask):
        # Get the color of the original object
        color = input_grid[x, y]
        
        # Find all symmetrical points
        for i in range(4):  # 4 rotations (0, 90, 180, 270 degrees)
            rotated_x, rotated_y = symmetries.apply(x, y, iters=i)
            if 0 <= rotated_x < input_grid.shape[0] and 0 <= rotated_y < input_grid.shape[1]:
                if input_grid[rotated_x, rotated_y]!= Color.BLACK:
                    input_grid[x, y] = input_grid[rotated_x, rotated_y]
                    break

    # Extract the filled-in region where the black rectangle was
    filled_in_region = np.full_like(input_grid, Color.BLACK)
    filled_in_region[black_rectangle_mask] = input_grid[black_rectangle_mask]
    return filled_in_region