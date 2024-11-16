from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern extraction, color mapping, grid transformation

# description:
# In the input, you will see a grid with a colored pattern along the border and a gray area in the center.
# To create the output, you should extract the pattern from the border and fill the gray area with the extracted pattern.
# The output grid should maintain the size of the original grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid for the output
    output_grid = np.copy(input_grid)

    # Find the bounding box of the non-background pixels
    bounding_box_coords = bounding_box(output_grid, background=Color.BLACK)
    x, y, width, height = bounding_box_coords

    # Extract the pattern from the border
    border_pattern = output_grid[x:x + width, y:y + height]

    # Crop the pattern to remove any excess black pixels
    pattern = crop(border_pattern, background=Color.BLACK)

    # Fill the gray area with the extracted pattern
    output_grid[x:x + pattern.shape[0], y:y + pattern.shape[1]] = pattern

    return output_grid