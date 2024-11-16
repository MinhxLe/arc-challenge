from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel expansion, color transformation

# description:
# In the input, you will see a small colored object surrounded by black pixels. 
# To make the output, expand the object outward by one pixel in all directions, 
# changing the color of the new border pixels to the color of the original object.

def transform(input_grid):
    # Plan:
    # 1. Find the color of the object (the non-black pixels).
    # 2. Find the coordinates of the object.
    # 3. Expand the object by one pixel in all directions.
    
    # 1. Detect the object (the non-black pixels)
    object_components = detect_objects(grid=input_grid, monochromatic=False, background=Color.BLACK, connectivity=4)
    
    # There should be exactly one object detected
    assert len(object_components) == 1
    object_component = object_components[0]

    # 2. Find the color of the object
    object_color = np.unique(object_component[object_component!= Color.BLACK])[0]

    # 3. Create an output grid
    output_grid = np.full((input_grid.shape[0] + 2, input_grid.shape[1] + 2), Color.BLACK)

    # Blit the original object into the center of the output grid
    blit_sprite(output_grid, object_component, x=1, y=1, background=Color.BLACK)

    # Expand the object by one pixel in all directions
    for x in range(1, output_grid.shape[0] - 1):
        for y in range(1, output_grid.shape[1] - 1):
            if output_grid[x, y] == object_color:
                # Change the surrounding pixels to the object's color
                output_grid[x - 1, y] = object_color  # Up
                output_grid[x + 1, y] = object_color  # Down
                output_grid[x, y - 1] = object_color  # Left
                output_grid[x, y + 1] = object_color  # Right
                output_grid[x - 1, y - 1] = object_color  # Top-Left
                output_grid[x - 1, y + 1] = object_color  # Top-Right
                output_grid[x + 1, y - 1] = object_color  # Bottom-Left
                output_grid[x + 1, y + 1] = object_color  # Bottom-Right

    return output_grid