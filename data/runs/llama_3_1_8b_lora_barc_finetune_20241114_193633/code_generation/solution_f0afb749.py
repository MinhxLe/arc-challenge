from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern duplication, rotation, color transformation

# description:
# In the input, you will see a small pattern of colored pixels in a 2x2 grid.
# To create the output, duplicate the pattern multiple times in a larger grid (e.g., 10x10), rotating each duplicate by 90 degrees.
# The output grid should contain the original pattern and its rotations.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Step 1: Find the connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4)

    # Step 2: Assuming there's only one pattern, get the first object
    pattern = objects[0]
    pattern_cropped = crop(pattern, background=Color.BLACK)

    # Step 3: Create a new output grid with the size of 10x10
    output_size = 10
    output_grid = np.full((output_size, output_size), Color.BLACK)

    # Step 4: Define the positions for the pattern and its rotations
    positions = [
        (0, 0),  # Top-left
        (0, 4),  # Top-right
        (4, 0),  # Bottom-left
        (4, 4)   # Bottom-right
    ]

    for idx, (x, y) in enumerate(positions):
        # Step 5: Rotate the pattern by 90 degrees for each position
        rotated_pattern = np.rot90(pattern_cropped, k=(idx % 4))
        blit_sprite(output_grid, rotated_pattern, x=x, y=y, background=Color.BLACK)

    return output_grid