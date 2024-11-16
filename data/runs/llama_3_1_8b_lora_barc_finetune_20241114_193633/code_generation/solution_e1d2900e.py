from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, filling, connectivity

# description:
# In the input, you will see a grid filled with a black background and colored pixels. 
# To make the output, replace each colored pixel with a new color based on the following mapping:
# red -> blue, blue -> green, green -> yellow, yellow -> red, 
# and all other colors remain unchanged. Pixels that are connected to the background (black) should not be changed.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    # Define the color mapping
    color_map = {
        Color.RED: Color.BLUE,
        Color.BLUE: Color.GREEN,
        Color.GREEN: Color.YELLOW,
        Color.YELLOW: Color.RED,
    }

    # Create a mask to identify background pixels
    background_mask = (input_grid == Color.BLACK)

    # Apply the color mapping while preserving background
    output_grid[~background_mask] = np.vectorize(lambda color: color_map.get(color, color))(output_grid[~background_mask])

    return output_grid