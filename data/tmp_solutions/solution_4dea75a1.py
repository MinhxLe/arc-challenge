from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern replication, color mapping, grid transformation

# description:
# In the input, you will see a small pattern of colored pixels in the top left corner of the grid.
# To create the output, replicate this pattern across the entire grid, ensuring that the original pattern's colors are preserved and arranged in a tiled manner.


def transform(input_grid):
    # Get the pattern from the input grid
    pattern = crop(input_grid)

    # Determine the size of the output grid
    output_height, output_width = input_grid.shape
    pattern_height, pattern_width = pattern.shape

    # Create an output grid filled with the background color
    output_grid = np.full((output_height, output_width), Color.BLACK)

    # Fill the output grid by replicating the pattern
    for i in range(0, output_height, pattern_height):
        for j in range(0, output_width, pattern_width):
            blit_sprite(output_grid, pattern, x=j, y=i, background=Color.BLACK)

    return output_grid

