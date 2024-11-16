from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, color replacement

# description:
# In the input you will see a grid with two colors: one color for the background and one color for the object. 
# The object is symmetric with respect to the center of the grid. To make the output, you should identify the color of the object,
# replace all pixels of the object with the background color, and then reflect the background across the center of the grid.

def transform(input_grid):
    # Create a copy of the input grid to work with
    output_grid = np.copy(input_grid)

    # Find the connected components of the object (non-background pixels)
    objects = find_connected_components(output_grid, background=Color.BLACK, monochromatic=True)

    # Get the color of the object
    object_color = None
    for obj in objects:
        object_color = obj[0][0]  # Assuming the object is monochromatic
        break

    # Replace the object color with the background color (black)
    output_grid[output_grid == object_color] = Color.BLACK

    # Reflect the background across the center of the grid
    center_x = output_grid.shape[0] // 2
    center_y = output_grid.shape[1] // 2

    for x in range(center_x):
        for y in range(center_y):
            # Reflect the background color across the center
            output_grid[center_x + (center_x - x), center_y + (center_y - y)] = Color.BLACK

    return output_grid