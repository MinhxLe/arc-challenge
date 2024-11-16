from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, filling, object detection

# description:
# In the input, you will see a grid with a colored shape surrounded by black pixels.
# To create the output, you should fill the area outside the shape with a specified color (e.g., red) while keeping the original shape intact.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid to work on
    output_grid = np.copy(input_grid)

    # Detect the object (the colored shape) in the input grid
    object_components = detect_objects(grid=input_grid, monochromatic=False, background=Color.BLACK, connectivity=4, allowed_dimensions=None)
    assert len(object_components) == 1  # Expecting only one object
    object_component = object_components[0]

    # Get the bounding box of the detected object
    x, y, width, height = bounding_box(object_component)

    # Fill the area outside the object with red color
    for i in range(output_grid.shape[0]):
        for j in range(output_grid.shape[1]):
            if output_grid[i, j] == Color.BLACK:  # If it's a background pixel
                # Check if it is within the bounds of the bounding box of the object
                if (i >= y and i < y + height) and (j >= x and j < x + width):
                    continue  # Skip the object area
                output_grid[i, j] = Color.RED  # Fill with red color

    return output_grid