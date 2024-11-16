from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel expansion, color transformation, connected components

# description:
# In the input, you will see a grid with a single colored object surrounded by black pixels. The object has a distinct color and can be of any shape.
# To create the output, expand the colored object outward by one pixel in all directions, filling the new pixels with a new color (the same color as the original object).

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find the connected components in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)

    # There should be exactly one colored object
    assert len(objects) == 1
    original_object = objects[0]

    # Get the color of the original object
    original_color = original_object[original_object!= Color.BLACK][0]

    # Create a new output grid filled with the background color
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Find the bounding box of the original object
    x, y, width, height = bounding_box(original_object, background=Color.BLACK)

    # Expand the object outward by one pixel in all directions
    for i in range(x - 1, x + width + 1):
        for j in range(y - 1, y + height + 1):
            if 0 <= i < output_grid.shape[0] and 0 <= j < output_grid.shape[1]:
                if output_grid[i, j] == Color.BLACK:  # Only expand into black pixels
                    output_grid[i, j] = original_color

    return output_grid