from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object detection, symmetry detection

# description:
# In the input, you will see a grid with various colored objects. To create the output, detect any objects that have horizontal or vertical symmetry
# and fill them with a new color while leaving the rest of the grid unchanged.

def transform(input_grid):
    # Detect objects in the grid
    objects = detect_objects(grid=input_grid, monochromatic=False, background=Color.BLACK, connectivity=4)
    output_grid = input_grid.copy()

    # Define the color to fill symmetrical objects with
    fill_color = Color.PURPLE

    for obj in objects:
        # Check for horizontal symmetry
        if np.array_equal(obj, np.flipud(obj)):
            # If horizontally symmetrical, fill with the new color
            x, y = object_position(obj, background=Color.BLACK, anchor="upper left")
            output_grid[x, y:y + obj.shape[1]] = fill_color
            output_grid[x + obj.shape[0] - 1, y:y + obj.shape[1]] = fill_color
        
        # Check for vertical symmetry
        if np.array_equal(obj, np.fliplr(obj)):
            # If vertically symmetrical, fill with the new color
            x, y = object_position(obj, background=Color.BLACK, anchor="upper left")
            output_grid[x:x + obj.shape[0], y] = fill_color
            output_grid[x:x + obj.shape[0], y + obj.shape[1] - 1] = fill_color

    return output_grid