from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color filling, bounding box detection

# description:
# In the input grid, you will see a colored shape surrounded by a background color (black).
# To create the output grid, you should fill the entire bounding box of the colored shape with a single color that is not black.
# The output grid should be the same size as the bounding box of the shape.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find the connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True, connectivity=4)

    # Assume there is only one object in the grid
    assert len(objects) == 1
    colored_shape = objects[0]

    # Get the bounding box of the colored shape
    x, y, width, height = bounding_box(colored_shape)

    # Create an output grid of the same size as the bounding box
    output_grid = np.full((width, height), Color.BLACK)

    # Get the color of the shape
    shape_color = np.unique(colored_shape)[0]

    # Fill the bounding box with the shape's color
    output_grid.fill(shape_color)

    return output_grid