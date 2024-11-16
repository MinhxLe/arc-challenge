from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, connectivity, color transformation

# description:
# The input consists of a grid filled with various colors, with a specific color marking a pathway.
# The output should create a new grid where:
# 1. All pixels that are part of the pathway are colored with a new color (the pathway color).
# 2. All other pixels in the grid should be transformed into a new color based on their original color:
#    - If the original color is black, it remains black.
#    - If the original color is any other color, it is replaced with a specific new color defined in the transformation.

def transform(input_grid):
    # Create a copy of the input grid for output
    output_grid = np.copy(input_grid)

    # Define the pathway color and the new color for the pathway
    pathway_color = Color.BLUE  # The pathway color
    new_color = Color.YELLOW   # The new color for other pixels

    # Find the pathway pixels
    pathway_mask = (output_grid == pathway_color)

    # Change pathway pixels to a new color
    output_grid[pathway_mask] = Color.YELLOW  # Change pathway color to yellow

    # Transform other pixels (not black and not pathway)
    for x in range(output_grid.shape[0]):
        for y in range(output_grid.shape[1]):
            if (output_grid[x, y]!= Color.BLACK) and (not pathway_mask[x, y]):
                # Change color to new_color if it's not black or pathway color
                output_grid[x, y] = new_color

    return output_grid