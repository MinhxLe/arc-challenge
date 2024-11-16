from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, boundary tracing, filling

# description:
# In the input, you will see a grid filled with various colored shapes, including a blue shape on a black background.
# To make the output, trace the boundary of the blue shape and fill the inside with a specified color (e.g., red).
# If the blue shape has any holes, they should remain black in the output.

def transform(input_grid):
    # Create a copy of the input grid for output
    output_grid = np.copy(input_grid)

    # Find the blue shape
    blue_objects = detect_objects(grid=input_grid, colors=[Color.BLUE], monochromatic=True, connectivity=4)

    # There should only be one blue shape in the input
    assert len(blue_objects) == 1
    blue_object = blue_objects[0]

    # Find the bounding box of the blue shape
    x, y, width, height = bounding_box(blue_object, background=Color.BLACK)
    
    # Get the interior of the blue shape (the area that will be filled)
    interior_mask = object_interior(blue_object, background=Color.BLACK, connectivity=4)

    # Fill the interior with red
    output_grid[interior_mask] = Color.RED

    # Draw the boundary of the blue shape
    for x in range(width):
        for y in range(height):
            if blue_object[x, y] == Color.BLUE:
                # Draw a border around the blue shape
                output_grid[x, y] = Color.BLUE

    return output_grid