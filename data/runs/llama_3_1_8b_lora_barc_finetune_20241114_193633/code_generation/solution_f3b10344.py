from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, object extraction, boundary detection

# description:
# In the input, you will see a grid filled with multiple colored objects, and a single black background.
# To create the output grid, extract all objects of a specific color (e.g., red) and map them to a new grid,
# changing their color to a different specified color (e.g., green). The output grid should retain the positions of
# the original objects but with their colors changed to the specified new color.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid for the output
    output_grid = np.copy(input_grid)

    # Detect all objects in the grid
    objects = detect_objects(grid=input_grid, monochromatic=True, background=Color.BLACK, connectivity=4, allowed_dimensions=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)])

    # Iterate through detected objects and change their colors
    for obj in objects:
        # Check if the object is red
        if np.any(obj == Color.RED):
            # Change color of the object to green
            output_grid[obj!= Color.BLACK] = Color.GREEN

    return output_grid