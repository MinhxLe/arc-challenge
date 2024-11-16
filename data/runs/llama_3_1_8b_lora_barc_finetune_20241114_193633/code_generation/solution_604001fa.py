from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, shape transformation, rotation

# description:
# In the input, you will see a grid containing several colored objects (circles and squares) on a black background.
# To create the output, transform the shapes as follows:
# - For each circle, change its color to green.
# - For each square, change its color to blue.
# Additionally, if a shape touches the boundary of the grid, it will be rotated by 90 degrees clockwise.

def transform(input_grid):
    # Create a copy of the input grid to manipulate
    output_grid = np.copy(input_grid)

    # Find all connected components (shapes) in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)

    for obj in objects:
        # Check if the object is a circle or square
        bounding_box_coords = bounding_box(obj, background=Color.BLACK)
        x, y, w, h = bounding_box_coords

        # Check if the object is a circle (aspect ratio)
        aspect_ratio = h / w
        if aspect_ratio < 0.5:  # Assume square is defined as having an aspect ratio less than 0.5
            # Change color to blue
            obj[obj!= Color.BLACK] = Color.BLUE
        else:  # Otherwise, it's a circle
            # Change color to green
            obj[obj!= Color.BLACK] = Color.GREEN

        # Check if it touches the boundary
        if np.any(obj[0, :]!= Color.BLACK) or np.any(obj[-1, :]!= Color.BLACK) or np.any(obj[:, 0]!= Color.BLACK) or np.any(obj[:, -1]!= Color.BLACK):
            # Rotate the shape 90 degrees clockwise
            # The rotation matrix for 90 degrees clockwise
            rotation_matrix = np.array([[0, -1], [1, 0]])
            # Apply the rotation
            rotated_shape = np.empty_like(obj)
            for i in range(obj.shape[0]):
                for j in range(obj.shape[1]):
                    rotated_shape[j, obj.shape[0] - 1 - i] = obj[i, j]
            # Blit the rotated shape onto the output grid
            blit_object(output_grid, rotated_shape, background=Color.BLACK)

    return output_grid