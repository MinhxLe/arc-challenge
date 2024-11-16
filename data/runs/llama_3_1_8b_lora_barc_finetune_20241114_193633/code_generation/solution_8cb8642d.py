from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# rotation, symmetry, color filling

# description:
# In the input, you will see a grid with a pattern that has rotational symmetry. 
# To make the output, fill the empty spaces around the pattern with the colors from the pattern, 
# maintaining the rotational symmetry.

def transform(input_grid):
    # Get the bounding box of the non-background pixels
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4)
    assert len(objects) > 0, "No pattern found in the input grid"
    
    # Assume the first object is the main pattern with rotational symmetry
    pattern = objects[0]
    x, y, w, h = bounding_box(pattern, background=Color.BLACK)

    # Create an output grid of the same size as the input grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Fill the output grid with the pattern colors while maintaining rotational symmetry
    for x_pos in range(h):
        for y_pos in range(w):
            if pattern[x_pos, y_pos]!= Color.BLACK:
                color = pattern[x_pos, y_pos]
                # Fill all 8 symmetrical positions
                output_grid[x, y] = color
                output_grid[x + 1, y] = color
                output_grid[x, y + 1] = color
                output_grid[x + 1, y + 1] = color
                output_grid[x, y - 1] = color
                output_grid[x + 1, y - 1] = color
                output_grid[x - 1, y] = color
                output_grid[x - 1, y + 1] = color
                output_grid[x - 1, y - 1] = color
                output_grid[x - 1, y] = color

    # Return the filled output grid
    return output_grid