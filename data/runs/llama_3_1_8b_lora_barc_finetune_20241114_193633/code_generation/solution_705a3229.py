from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# radial symmetry, color mapping

# description:
# In the input, you will see a colored object in the center of the grid surrounded by a black background.
# To make the output, create a radial pattern of the same color as the object, extending outward from the center of the object until it reaches the edge of the grid.
# The radial pattern should be symmetrical and should fill the entire grid.

def transform(input_grid):
    # Step 1: Detect the colored object in the center of the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)
    
    # There should be only one colored object in the center
    assert len(objects) == 1
    center_object = objects[0]
    
    # Step 2: Get the bounding box of the colored object
    x, y, width, height = bounding_box(center_object, background=Color.BLACK)
    
    # Step 3: Get the color of the object
    color = center_object[center_object!= Color.BLACK][0]
    
    # Step 4: Create the output grid
    output_grid = np.full(input_grid.shape, Color.BLACK)
    
    # Step 5: Fill the output grid with the radial pattern
    for i in range(output_grid.shape[0]):
        for j in range(output_grid.shape[1]):
            # Calculate the distance from the center of the object
            distance = int(np.sqrt((i - y) ** 2 + (j - x) ** 2))
            # If within the bounding box of the object, fill it with the object's color
            if distance < max(width, height):
                output_grid[i, j] = color
    
    return output_grid