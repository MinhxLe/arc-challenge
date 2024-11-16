from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, pattern recognition, object detection

# description:
# In the input, you will see a grid filled with different colored shapes. Each shape is a connected component of the same color. 
# The task is to find the largest shape and fill the entire grid with the color of that shape, while making all other colors transparent.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find all connected components in the input grid
    objects = find_connected_components(input_grid, monochromatic=True, connectivity=8)

    # Identify the largest shape
    largest_shape = max(objects, key=lambda obj: np.sum(obj!= Color.BLACK))

    # Get the color of the largest shape
    largest_color = largest_shape[largest_shape!= Color.BLACK][0]

    # Create the output grid filled with the largest shape's color
    output_grid = np.full(input_grid.shape, Color.BLACK)
    output_grid[largest_shape!= Color.BLACK] = largest_color

    return output_grid