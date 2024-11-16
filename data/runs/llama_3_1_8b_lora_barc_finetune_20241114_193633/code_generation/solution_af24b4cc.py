from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern recognition, color mapping, grid transformation

# description:
# In the input, you will see a grid containing multiple 2x2 patterns of colors arranged in a 3x3 grid.
# Each pattern corresponds to a specific color. The output should be a 3x3 grid where each cell is filled
# with the corresponding color from the input patterns based on their positions.
# The output grid should be centered in the original grid.

def transform(input_grid):
    # Find all the 2x2 patterns in the input grid
    patterns = detect_objects(grid=input_grid, allowed_dimensions=[(2, 2), (2, 2)],
                             monochromatic=True, background=Color.BLACK, connectivity=4)

    # Prepare the output grid, which will be a 3x3 grid
    output_grid = np.full((3, 3), Color.BLACK)

    # Map the patterns to the output grid based on their positions
    for idx, pattern in enumerate(patterns):
        # Get the bounding box of the detected pattern
        x, y, w, h = bounding_box(pattern)
        
        # Extract the color from the pattern
        color = pattern[x, y]  # Assuming the pattern is monochromatic

        # Calculate the position in the output grid
        output_x = (idx % 3) * 3
        output_y = (idx // 3) * 3

        # Blit the color into the output grid
        output_grid = blit_sprite(output_grid, np.full((3, 3), color, dtype=int), x=output_x, y=output_y, background=Color.BLACK)

    return output_grid