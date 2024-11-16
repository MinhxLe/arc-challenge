from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color transformation, pattern extraction

# description:
# In the input, you will see a grid containing a 2x2 pattern of colors scattered throughout.
# To create the output, extract all 2x2 patterns from the input grid and convert them into a 1x3 vertical strip.
# Each extracted pattern will be colored based on the color of the top-left pixel of the original pattern.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Detect all 2x2 patterns in the input grid
    objects = detect_objects(grid=input_grid, allowed_dimensions=[(2, 2), (2, 2), (1, 2), (2, 1), (1, 1)], monochromatic=False, background=Color.BLACK)

    # Prepare the output grid
    output_grid = np.full((len(objects), 3), Color.BLACK, dtype=int)

    # Fill the output grid with the transformed patterns
    for index, obj in enumerate(objects):
        # Get the bounding box of the detected object
        x, y, w, h = bounding_box(obj)
        # Crop the pattern
        pattern = crop(obj, background=Color.BLACK)

        # Determine the color of the top-left pixel of the pattern
        color = pattern[0, 0]

        # Place the transformed pattern in the output grid
        for i in range(3):
            for j in range(3):
                if i < 3 and j < 3:
                    output_grid[index, j] = color  # Fill the corresponding row with the pattern's color

    return output_grid