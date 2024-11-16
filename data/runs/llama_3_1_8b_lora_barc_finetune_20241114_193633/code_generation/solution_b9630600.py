from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color counting, boundary detection, filling

# description:
# In the input, you will see a grid with a black background and a colored object on a green background.
# The object has a distinct color. To create the output, count the number of pixels of that color and fill the entire output grid with that color, 
# while leaving the black background intact.

def transform(input_grid):
    # Step 1: Count the number of pixels of the object's color (non-black)
    object_color = None
    object_count = 0
    
    # Find the color of the object (non-black)
    for color in np.unique(input_grid):
        if color!= Color.BLACK:
            object_color = color
            break
    
    # Count the number of pixels of the object's color
    object_count = np.sum(input_grid == object_color)
    
    # Step 2: Create the output grid filled with the object's color
    output_grid = np.full(input_grid.shape, Color.BLACK)
    output_grid.fill(object_color)

    return output_grid