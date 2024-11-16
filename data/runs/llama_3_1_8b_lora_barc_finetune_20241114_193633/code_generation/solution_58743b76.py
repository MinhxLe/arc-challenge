from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, pixel transformation, symmetry

# description:
# The input grid consists of a symmetric pattern of colored pixels with a black background. 
# The output grid should transform the colors of the pixels in the pattern based on a color mapping:
# blue -> yellow, green -> blue, red -> pink, and black remains black. 
# Additionally, the grid should be mirrored horizontally before applying the color mapping.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid for the output
    output_grid = np.copy(input_grid)

    # Create a color mapping
    color_map = {
        Color.BLUE: Color.YELLOW,
        Color.GREEN: Color.BLUE,
        Color.RED: Color.PINK,
        Color.BLACK: Color.BLACK
    }

    # Mirror the grid horizontally
    mirrored_grid = np.flip(output_grid, axis=1)

    # Apply the color mapping
    for original_color, new_color in color_map.items():
        output_grid[mirrored_grid == original_color] = new_color

    return output_grid