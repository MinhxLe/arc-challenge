from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color propagation, boundary detection, connectivity

# description:
# In the input, you will see a grid with colored pixels, some of which are connected. 
# To create the output, propagate the colors of the connected components into the adjacent cells that are currently black. 
# The propagation should only occur in the cardinal directions (up, down, left, right) and not through black pixels.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    # Find all connected components in the grid
    components = find_connected_components(input_grid, background=Color.BLACK, monochromatic=False, connectivity=4)

    # For each connected component, propagate its color to adjacent black pixels
    for component in components:
        color = np.unique(component[component!= Color.BLACK])[0]  # Get the color of the component
        for x, y in np.argwhere(component!= Color.BLACK):
            # Check all 4 cardinal directions
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < output_grid.shape[0] and 0 <= ny < output_grid.shape[1]:
                    if output_grid[nx, ny] == Color.BLACK:
                        output_grid[nx, ny] = color

    return output_grid