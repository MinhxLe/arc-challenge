from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# shape detection, color mapping, bounding box extraction

# description:
# In the input grid, you will see several colored shapes on a black background. 
# The task is to identify each shape, determine its color, and fill the entire area of the shape 
# with its respective color in the output grid. The output grid should maintain the same dimensions as the input grid.

def transform(input_grid):
    # Find all connected components (shapes) in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=8, monochromatic=False)

    # Create an output grid initialized to black
    output_grid = np.full(input_grid.shape, Color.BLACK)

    for obj in objects:
        # Get the bounding box of the shape
        x, y, width, height = bounding_box(obj, background=Color.BLACK)
        
        # Crop the shape from the grid
        shape = crop(obj, background=Color.BLACK)

        # Get the color of the shape (the color of the top-left pixel)
        color = shape[0, 0]

        # Fill the output grid with the shape color
        output_grid[x:x+height, y:y+width] = color

    return output_grid