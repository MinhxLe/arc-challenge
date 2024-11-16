from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object detection, color mapping, boundary detection

# description:
# In the input, you will see a grid containing a colored object surrounded by a black background.
# The goal is to detect the boundary of the object and replace the interior of the object with a new color, while keeping the boundary color intact.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create an output grid initialized to the background color
    output_grid = np.copy(input_grid)

    # Detect the boundary of the colored object
    boundary_mask = (output_grid!= Color.BLACK)
    
    # Find the connected components of the object
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)

    # Assume there is only one object, and we will modify it
    if len(objects) == 0:
        return output_grid  # No object detected, return the original grid

    # Assuming the first detected object is the one we want to modify
    main_object = objects[0]
    
    # Get the bounding box of the main object
    x, y, width, height = bounding_box(main_object, background=Color.BLACK)
    
    # Replace the interior of the object with a new color (e.g., Color.GREEN)
    new_color = Color.GREEN
    for i in range(x, x + width):
        for j in range(y, y + height):
            if (i >= 0 and i < output_grid.shape[0] and j >= 0 and j < output_grid.shape[1] and
                main_object[i - x, j - y]!= Color.BLACK and
                (i < x or i >= x + width or j < y or j >= y + height)):
                output_grid[i, j] = new_color

    return output_grid