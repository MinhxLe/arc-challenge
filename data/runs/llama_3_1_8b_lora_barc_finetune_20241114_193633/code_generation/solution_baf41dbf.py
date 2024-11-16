from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel expansion, boundary detection, shape manipulation

# description:
# In the input, you will see a colored object surrounded by a black background. To make the output, 
# expand the object outward by one pixel in all directions, filling in the new pixels with the color of the original object.

def transform(input_grid):
    # Find the connected components in the input grid
    objects = find_connected_components(input_grid, monochromatic=True, connectivity=4)
    
    # Assuming there is only one object in the input
    assert len(objects) == 1
    original_object = objects[0]

    # Get the color of the original object
    object_color = np.unique(original_object[original_object!= Color.BLACK])[0]

    # Create an output grid initialized to black
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Get the bounding box of the original object
    x, y, width, height = bounding_box(original_object, background=Color.BLACK)

    # Expand the object by one pixel in all directions
    for i in range(x - 1, x + width + 1):
        for j in range(y - 1, y + height + 1):
            if (0 <= i < output_grid.shape[0]) and (0 <= j < output_grid.shape[1]):
                if (x <= i < x + width) and (y <= j < y + height):
                    output_grid[i, j] = object_color
                else:
                    output_grid[i, j] = Color.BLACK

    return output_grid