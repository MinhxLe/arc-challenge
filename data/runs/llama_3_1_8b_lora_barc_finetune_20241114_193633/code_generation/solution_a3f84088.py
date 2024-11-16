from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# boundary detection, color filling, connectivity

# description:
# In the input, you will see a grid with a colored object surrounded by a boundary of black pixels. 
# To create the output, fill the area inside the boundary with a different color, effectively creating a filled region.

def transform(input_grid):
    # Create a copy of the input grid for output
    output_grid = np.copy(input_grid)
    
    # Find the background color
    background_color = Color.BLACK
    
    # Detect the colored object
    objects = detect_objects(grid=input_grid, colors=Color.NOT_BLACK, monochromatic=True, connectivity=4, background=background_color)

    # There should only be one colored object
    assert len(objects) == 1
    colored_object = objects[0]

    # Get the bounding box of the colored object
    x, y, w, h = bounding_box(colored_object, background=background_color)

    # Fill the area inside the bounding box with a new color
    fill_color = Color.RED  # You can choose any color for filling
    for i in range(x, x + w):
        for j in range(y, y + h):
            if colored_object[i - x, j - y]!= background_color:
                output_grid[i, j] = fill_color

    return output_grid