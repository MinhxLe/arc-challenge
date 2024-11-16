from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color replacement, pattern mirroring, grid transformation

# description:
# In the input, you will see a grid with a central pattern and a border of colors around it.
# To make the output, mirror the central pattern across the border colors in a specified direction
# while keeping the border colors intact.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid for the output
    output_grid = np.copy(input_grid)

    # Detect the bounding box of the central pattern (non-background pixels)
    central_object = detect_objects(grid=input_grid, background=Color.BLACK, monochromatic=False, connectivity=4, allowed_dimensions=None, colors=None, can_overlap=False)[0]
    x, y, w, h = bounding_box(central_object, background=Color.BLACK)

    # Crop the central pattern
    central_pattern = crop(central_object, background=Color.BLACK)

    # Get the border colors
    top_border = output_grid[0, :w]
    left_border = output_grid[:h, 0]
    right_border = output_grid[:h, -w]
    bottom_border = output_grid[-1, :w]

    # Create a mirrored pattern
    mirrored_pattern = np.flipud(central_pattern)

    # Replace the borders with the mirrored pattern
    output_grid[0, :w] = top_border
    output_grid[:h, 0] = left_border
    output_grid[-1, :w] = bottom_border
    output_grid[:h, -w] = right_border

    # Place the mirrored pattern in the center
    output_grid[1:-1, 1:-1] = mirrored_pattern

    return output_grid