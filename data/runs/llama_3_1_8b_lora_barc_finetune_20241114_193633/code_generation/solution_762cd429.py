from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color gradient, expansion

# description:
# In the input, you will see a grid with a central colored object and a black background. The object is surrounded by a color gradient that fades from the object's color to the black background. 
# To make the output, expand the object outward while maintaining the gradient pattern until it reaches the edge of the grid or hits another color.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Plan:
    # 1. Detect the central object in the grid.
    # 2. Determine the color of the object.
    # 3. Create a gradient from the object's color to black, expanding the object outward until it hits the edge of the grid or another color.

    # Detect the central object in the grid
    objects = detect_objects(input_grid, monochromatic=False, background=Color.BLACK, connectivity=4, allowed_dimensions=[(1, 1)], predicate=lambda obj: np.any(obj!= Color.BLACK))
    assert len(objects) == 1
    central_object = objects[0]

    # Get the color of the object
    object_color = np.unique(central_object[central_object!= Color.BLACK])[0]

    # Create the output grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Find the bounding box of the central object
    x, y, width, height = bounding_box(central_object)

    # Expand the object outward until it hits the edge or another color
    for i in range(input_grid.shape[0]):
        for j in range(input_grid.shape[1]):
            if output_grid[i, j] == Color.BLACK:
                # Check if this pixel is part of the object
                if (i >= x - width // 2 and i < x + width // 2 and
                    j >= y - height // 2 and j < y + height // 2):
                    output_grid[i, j] = object_color
                elif (i >= x and i < x + width and
                        j >= y and j < y + height and
                        input_grid[i, j] == object_color and
                        output_grid[i, j] == Color.BLACK):
                    # Expand the object
                    output_grid[i, j] = object_color

    return output_grid