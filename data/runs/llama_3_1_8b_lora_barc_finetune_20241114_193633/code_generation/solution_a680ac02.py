from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object detection, bounding box extraction, color transformation

# description:
# In the input, you will see a grid with various colored shapes scattered on a black background. 
# The output should be a grid where each object's color is changed to match the color of the first shape detected in the grid. 
# If no objects are detected, the output should remain unchanged.

def transform(input_grid):
    # Detect all colored objects in the input grid
    objects = detect_objects(grid=input_grid, monochromatic=False, background=Color.BLACK, connectivity=4, allowed_dimensions=None, colors=None, can_overlap=False)
    
    # Check if any objects were detected
    if not objects:
        # If no objects detected, return the grid unchanged
        return input_grid

    # Get the color of the first detected object
    first_object_color = np.unique(objects[0][objects[0]!= Color.BLACK])[0]

    # Create an output grid initialized to the input grid
    output_grid = np.copy(input_grid)

    # Change the color of each detected object to the color of the first object
    for obj in objects:
        # Change the color of the object to the first detected color
        output_grid[obj!= Color.BLACK] = first_object_color

    return output_grid