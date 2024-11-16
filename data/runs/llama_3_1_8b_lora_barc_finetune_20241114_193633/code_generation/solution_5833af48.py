from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern extraction, mirroring, grid transformation

# description:
# In the input, you will see a grid with a central pattern surrounded by colored pixels. 
# To create the output, you should extract the central pattern and mirror it across the grid's center, 
# filling the empty spaces with the mirrored pattern.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Crop the central pattern from the input grid
    central_pattern = crop(input_grid, background=Color.BLACK)

    # Determine the dimensions of the central pattern
    pattern_height, pattern_width = central_pattern.shape

    # Create an output grid that is twice the size of the central pattern
    output_grid = np.full((pattern_height * 2, pattern_width * 2), Color.BLACK)

    # Place the original pattern in the top-left corner
    output_grid[:pattern_height, :pattern_width] = central_pattern

    # Mirror the pattern to the top-right corner
    output_grid[:pattern_height, pattern_width:] = np.flip(central_pattern, axis=1)

    # Mirror the pattern to the bottom-left corner
    output_grid[pattern_height:, :pattern_width] = np.flip(central_pattern, axis=0)

    # Mirror the pattern to the bottom-right corner
    output_grid[pattern_height:, pattern_width:] = np.flip(central_pattern)

    return output_grid