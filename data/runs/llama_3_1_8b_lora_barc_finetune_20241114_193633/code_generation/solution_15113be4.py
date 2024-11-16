from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel manipulation, color replacement

# description:
# In the input, you will see a grid with a black background and colored pixels scattered throughout. 
# The grid contains a specific color (e.g., blue) that forms a continuous shape, and the rest of the pixels are black.
# To make the output, you should replace the blue shape with a new color (e.g., green) while leaving the black background unchanged.
# Additionally, if the blue shape is surrounded by black pixels, you should add a border of a different color (e.g., orange) around the shape.

def transform(input_grid):
    # Create an output grid that starts as a copy of the input grid
    output_grid = np.copy(input_grid)

    # Find all connected components of blue pixels
    blue_objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True, connectivity=4)

    for obj in blue_objects:
        # Get the bounding box of the blue object
        x, y, width, height = bounding_box(obj, background=Color.BLACK)

        # Replace the blue shape with green pixels
        output_grid[x:x + width, y:y + height] = Color.GREEN

        # Check if the blue object is surrounded by black pixels (no black border around it)
        if np.all((x > 0) & (x + width - 1 < output_grid.shape[0] - 1) & 
                   (y > 0) & (y + height - 1 < output_grid.shape[1] - 1) & 
                   (output_grid[x-1:x+width+1, y-1:y+height+1] == Color.BLACK)):
            # Add a border of orange pixels around the shape
            output_grid[x-1:x+width+1, y-1:y+height+1] = Color.ORANGE

    return output_grid