from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# rotation, color mapping

# description:
# In the input, you will see a grid with a colored object on a black background. The object can be any shape,
# but it will be surrounded by a black border. To create the output, rotate the object 90 degrees clockwise
# and color all the pixels that are now black with yellow, while keeping the rest of the object's color.

def transform(input_grid):
    # Get the bounding box of the non-background pixels
    x, y, width, height = bounding_box(input_grid, background=Color.BLACK)
    
    # Extract the object from the input grid
    object_to_rotate = input_grid[x:x + width, y:y + height]

    # Rotate the object 90 degrees clockwise
    rotated_object = np.rot90(object_to_rotate, k=-1)

    # Create an output grid with the same shape as the input
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Place the original object and the rotated object in the output grid
    output_grid[x:x + width, y:y + height] = object_to_rotate
    output_grid[x:x + width, y + height:y + 2 * height] = np.where(rotated_object!= Color.BLACK, rotated_object, Color.YELLOW)

    return output_grid