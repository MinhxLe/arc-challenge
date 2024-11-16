from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, pattern replication, color replacement

# description:
# In the input, you will see a grid containing several patterns of different colors arranged in a row. 
# The first row contains the colors to be copied to the output grid. 
# The second row contains the corresponding patterns. 
# To create the output, replicate the patterns in the second row according to the colors in the first row, 
# replacing the colors in the output grid with the colors from the first row.

def transform(input_grid):
    # Get the first row colors
    first_row_colors = input_grid[0]

    # Create a mapping from the first row colors to the second row patterns
    pattern_mapping = {}
    for color in first_row_colors:
        # Find the corresponding pattern in the second row
        pattern_row = input_grid[1]
        color_pattern = pattern_row[np.where(pattern_row == color)[0][0]]
        pattern_mapping[color] = color_pattern

    # Create the output grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Replicate patterns based on the first row colors
    for x, color in enumerate(first_row_colors):
        if color in pattern_mapping:
            pattern = pattern_mapping[color]
            # Calculate the position to place the pattern in the output grid
            output_grid[1:, x] = pattern

    return output_grid