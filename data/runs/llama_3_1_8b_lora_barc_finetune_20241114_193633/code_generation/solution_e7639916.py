from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# boundary tracing, filling

# description:
# In the input, you will see a colored object on a black background. 
# To make the output, trace the boundary of the object and fill the area inside the boundary with a contrasting color while keeping the original color of the object intact.

def transform(input_grid):
    # Detect the object in the grid
    object_mask = (input_grid!= Color.BLACK)

    # Find the boundary of the object
    boundary_mask = object_boundary(object_mask, background=Color.BLACK)

    # Create a new output grid initialized to the background color
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Fill the area inside the boundary with a contrasting color (e.g., Color.BLUE)
    fill_color = Color.BLUE

    # Fill the inside of the boundary with the contrasting color
    for x, y in np.argwhere(object_mask):
        if boundary_mask[x, y]:
            # Check if the pixel is part of the boundary
            # If it is, we fill the inside with the contrasting color
            if object_mask[x-1, y] and x > 0:  # Top
                output_grid[x-1, y] = fill_color
            if object_mask[x+1, y] and x < output_grid.shape[0] - 1:  # Bottom
                output_grid[x+1, y] = fill_color
            if object_mask[x, y-1] and y > 0:  # Left
                output_grid[x, y-1] = fill_color
            if object_mask[x, y+1] and y < output_grid.shape[1] - 1:  # Right
                output_grid[x, y+1] = fill_color

    # Copy the original object color back to the output grid
    output_grid[object_mask] = input_grid[object_mask]

    return output_grid