from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object detection, color transformation, pixel manipulation

# description:
# In the input, you will see several objects of different colors on a black background.
# To create the output, you should identify the largest object and change its color to a specified target color while leaving the other objects unchanged.

def transform(input_grid):
    # Copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # Detect all connected components in the input grid
    objects = find_connected_components(grid=input_grid, background=Color.BLACK, connectivity=8)

    # Identify the largest object by the number of pixels
    largest_object = max(objects, key=lambda obj: np.sum(obj!= Color.BLACK), default=None)

    # If a largest object exists, change its color
    if largest_object is not None:
        target_color = Color.GREEN  # The target color we want to apply
        largest_object[largest_object!= Color.BLACK] = target_color  # Change its color to the target color

    return output_grid