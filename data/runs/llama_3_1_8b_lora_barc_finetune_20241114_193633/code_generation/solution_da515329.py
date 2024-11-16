from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel expansion, shape transformation

# description:
# In the input, you will see a small shape made of colored pixels. 
# To create the output, expand the shape by duplicating its pixels outward in all directions 
# until it fills the grid, maintaining its original color.

def transform(input_grid):
    # Detect the connected component (the shape) in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)
    
    # Assuming there is only one shape to expand
    shape = objects[0]

    # Get the color of the shape
    shape_color = shape[shape!= Color.BLACK][0]

    # Create the output grid, initialized to black
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Get the bounding box of the shape
    x, y, width, height = bounding_box(shape, background=Color.BLACK)

    # Fill the output grid with the expanded shape
    for i in range(height):
        for j in range(width):
            if shape[i, j]!= Color.BLACK:  # Check if the pixel is part of the shape
                # Calculate the expanded positions
                for dx in range(-i, i + 1):  # Expand in all 8 directions
                    for dy in range(-j, j + 1):
                        if 0 <= x + dx < output_grid.shape[0] and 0 <= y + dy < output_grid.shape[1]:
                            output_grid[x + dx, y + dy] = shape_color

    return output_grid