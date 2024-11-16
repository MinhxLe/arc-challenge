from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# connectivity, color tracing, pixel manipulation

# description:
# In the input, you will see a grid with a black background and colored pixels representing different regions.
# The task is to trace the outer boundary of each region and create a new grid that only contains the boundary pixels.
# Each boundary pixel should be colored based on the color of the region it belongs to.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a new grid to hold the boundary pixels
    boundary_grid = np.full(input_grid.shape, Color.BLACK)

    # Find connected components in the grid
    components = find_connected_components(input_grid, background=Color.BLACK, connectivity=4)

    for component in components:
        # Get the color of the current component
        color = component[0, 0]  # Assume the component is monochromatic
        # Create a mask for the current component
        mask = (component == color)

        # Create the boundary mask
        boundary_mask = object_boundary(mask, background=Color.BLACK)

        # Blit the boundary pixels onto the boundary_grid
        for x, y in np.argwhere(boundary_mask):
            boundary_grid[x, y] = color

    return boundary_grid