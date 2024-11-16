from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# rotation, color transformation, surrounding

# description:
# In the input, you will see a grid with colored pixels, some of which are red, blue, or green.
# To create the output grid, you should rotate the grid 90 degrees clockwise and then change the colors of the pixels based on the following mapping:
# red -> blue, blue -> green, green -> yellow, yellow -> red.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Rotate the grid 90 degrees clockwise
    rotated_grid = np.rot90(input_grid, k=-1)

    # Create a color mapping
    color_mapping = {
        Color.RED: Color.BLUE,
        Color.BLUE: Color.GREEN,
        Color.GREEN: Color.YELLOW,
        Color.YELLOW: Color.RED
    }

    # Apply color mapping
    output_grid = np.copy(rotated_grid)
    for color in Color.NOT_BLACK:
        if color in color_mapping:
            output_grid[rotated_grid == color] = color_mapping[color]

    return output_grid