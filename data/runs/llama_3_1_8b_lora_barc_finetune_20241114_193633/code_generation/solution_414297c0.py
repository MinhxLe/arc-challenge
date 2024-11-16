from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color separation, shape extraction

# description:
# In the input, you will see a grid filled with various colored shapes on a black background.
# To create the output, you need to extract all shapes that are the same color as the largest shape
# and place them in a new grid, while turning the rest of the grid to black.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find all connected components in the input grid
    objects = find_connected_components(input_grid, monochromatic=False, connectivity=4, background=Color.BLACK)
    
    # Identify the largest object by area (number of pixels)
    largest_object = max(objects, key=lambda obj: np.sum(obj!= Color.BLACK), default=None)

    # If no object found, return a grid filled with black
    if largest_object is None:
        return np.full(input_grid.shape, Color.BLACK)

    # Create a new output grid filled with black
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Place the largest object in the output grid
    for x, y in np.argwhere(largest_object!= Color.BLACK):
        output_grid[x, y] = largest_object[x, y]

    return output_grid