from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, color transformation

# description:
# In the input, you will see a grid with colored pixels forming a pattern that has rotational symmetry.
# To create the output, extract the pattern from the grid, rotate it 90 degrees clockwise, and place it back into the grid, 
# replacing the original pattern with the rotated version.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Step 1: Find the connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=8, monochromatic=True)
    
    # Step 2: Assume there is only one object, extract it
    if len(objects)!= 1:
        raise ValueError("Expected exactly one connected component in the input grid.")
    original_pattern = objects[0]

    # Step 3: Rotate the pattern 90 degrees clockwise
    rotated_pattern = np.rot90(original_pattern, k=-1)  # k=-1 for clockwise rotation

    # Step 4: Create output grid with the same shape as input
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Step 5: Blit the rotated pattern onto the output grid
    # We need to find the position to place the rotated pattern
    x, y = object_position(original_pattern, background=Color.BLACK, anchor="upper left")
    output_grid = blit_sprite(output_grid, rotated_pattern, x=x, y=y, background=Color.BLACK)

    return output_grid