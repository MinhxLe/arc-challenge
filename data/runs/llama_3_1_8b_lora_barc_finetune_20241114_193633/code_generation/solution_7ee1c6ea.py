from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, pixel transformation

# description:
# In the input, you will see a grid with a colored object and a border of a different color. 
# The goal is to create an output grid where the object is blended with the border color. 
# The blending should occur where the object meets the border, resulting in a gradient effect where the colors mix.

def transform(input_grid):
    # Detect the color of the border (the outermost layer)
    border_color = input_grid[0, 0]
    
    # Crop the object from the input grid
    object_mask = input_grid!= Color.BLACK
    object_pixels = np.argwhere(object_mask)

    # Create the output grid
    output_grid = np.copy(input_grid)

    # Blend the colors where the object meets the border
    for x, y in object_pixels:
        if input_grid[x, y]!= Color.BLACK:  # Only blend if it's not the background
            # Get the color of the pixel in the object
            object_color = input_grid[x, y]

            # Blend the colors
            # The blending is done by averaging the colors
            if x == 0 or x == input_grid.shape[0] - 1 or y == 0 or y == input_grid.shape[1] - 1:
                # If it's on the border, blend with the border color
                output_grid[x, y] = Color.BROWN  # Using brown as a blend result for simplicity
            else:
                # If not on the border, keep the original object color

    return output_grid