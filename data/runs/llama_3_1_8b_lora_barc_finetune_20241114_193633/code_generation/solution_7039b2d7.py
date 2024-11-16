from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, symmetry, reflection, bounding box

# description:
# In the input, you will see a grid filled with various colors. The grid has a black background. 
# The task is to find the largest connected component of the same color and reflect it across the center of the grid.
# The output grid should have the same dimensions as the input grid, but only the reflected component should be visible.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find all connected components in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)

    # Find the largest connected component
    largest_object = max(objects, key=lambda obj: np.sum(obj!= Color.BLACK))

    # Determine the color of the largest component
    largest_color = largest_object[0][0]

    # Create a new output grid filled with black
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Reflect the largest component across the center
    reflected_object = largest_object[::-1, ::-1]  # Reflect both horizontally and vertically

    # Calculate the center of the grid
    center_x = input_grid.shape[0] // 2
    center_y = input_grid.shape[1] // 2

    # Blit the reflected object onto the output grid
    for x in range(reflected_object.shape[0]):
        for y in range(reflected_object.shape[1]):
            if reflected_object[x, y]!= Color.BLACK:
                output_grid[center_x + x, center_y + y] = reflected_object[x, y]

    return output_grid