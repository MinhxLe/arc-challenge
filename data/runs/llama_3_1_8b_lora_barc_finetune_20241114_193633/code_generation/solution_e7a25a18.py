from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, border detection, color replacement

# description:
# In the input, you will see a grid with a single colored object surrounded by a black border. 
# To create the output, extract the object and replace all the black pixels in the grid with the color of the object, effectively creating a colored border around it.

def transform(input_grid):
    # Create a copy of the input grid to modify for output
    output_grid = np.copy(input_grid)

    # Detect the object in the grid (the non-black area)
    object_pixels = np.argwhere(input_grid!= Color.BLACK)
    
    # Check if there are any colored pixels
    if object_pixels.size == 0:
        return output_grid  # If no object found, return the original grid

    # Get the color of the object
    object_color = input_grid[object_pixels[0][0], object_pixels[0][1]]

    # Replace all black pixels with the object's color
    output_grid[output_grid == Color.BLACK] = object_color

    # Crop the output grid to the smallest bounding box around the colored pixels
    output_grid = crop(output_grid, background=Color.BLACK)

    return output_grid