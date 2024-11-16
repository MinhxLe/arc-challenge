from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# mirroring, reflection, color transformation

# description:
# In the input, you will see a grid with a series of colored pixels forming a shape along one edge (top, bottom, left, or right).
# To create the output grid, reflect the shape across the opposite edge (top to bottom, bottom to top, left to right, or right to left) 
# and fill in the mirrored area with the same color as the original shape.

def transform(input_grid):
    # Create an output grid initialized to black
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Detect the shape along the edge of the grid
    objects = detect_objects(grid=input_grid, monochromatic=False, background=Color.BLACK, connectivity=4, allowed_dimensions=[(1, 1)])

    # Assuming there's only one shape detected
    if objects:
        original_shape = objects[0]
        color = np.unique(original_shape[original_shape!= Color.BLACK])[0]

        # Find the bounding box of the shape
        x, y, w, h = bounding_box(original_shape, background=Color.BLACK)

        # Determine the edge along which to reflect
        if y == 0:  # Top edge
            reflection_edge = 'bottom'
            # Reflect vertically
            for i in range(h):
                for j in range(w):
                    if original_shape[i, j] == color:
                        output_grid[input_grid.shape[0] - 1 - i, j] = color
        elif x == 0:  # Left edge
            reflection_edge = 'right'
            # Reflect horizontally
            for i in range(h):
                for j in range(w):
                    if original_shape[i, j] == color:
                        output_grid[i, input_grid.shape[1] - 1 - j] = color
        elif y == input_grid.shape[1] - 1:  # Bottom edge
            reflection_edge = 'top'
            # Reflect vertically
            for i in range(h):
                for j in range(w):
                    if original_shape[i, j] == color:
                        output_grid[input_grid.shape[0] - 1 - i, j] = color
        else:  # Right edge
            reflection_edge = 'left'
            # Reflect horizontally
            for i in range(h):
                for j in range(w):
                    if original_shape[i, j] == color:
                        output_grid[i, input_grid.shape[1] - 1 - j] = color

    return output_grid