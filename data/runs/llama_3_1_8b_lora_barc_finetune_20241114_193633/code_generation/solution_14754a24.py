from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# connectivity, color transformation

# description:
# In the input, you will see a grid filled with colored pixels on a black background. 
# Each pixel can either be part of a connected component or a distinct colored pixel. 
# To create the output, you need to transform each connected component (group of adjacent colored pixels) into a new color.
# The transformation rules are:
# 1. If a component has more than 3 pixels, change its color to red.
# 2. If a component has 3 or fewer pixels, change its color to blue.
# The output should preserve the original colors of the distinct pixels.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a blank output grid
    output_grid = np.zeros_like(input_grid)

    # Detect connected components in the input grid
    components = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)

    # Transform each component based on the number of pixels
    for component in components:
        if np.sum(component!= Color.BLACK) > 3:
            # Change to red if the component has more than 3 pixels
            output_grid[output_grid == component[0, 0]] = Color.RED
        else:
            # Change to blue if the component has 3 or fewer pixels
            output_grid[output_grid == component[0, 0]] = Color.BLUE

    # Copy over the original colors of distinct pixels
    for x, y in np.argwhere(input_grid!= Color.BLACK):
        output_grid[x, y] = input_grid[x, y]

    return output_grid