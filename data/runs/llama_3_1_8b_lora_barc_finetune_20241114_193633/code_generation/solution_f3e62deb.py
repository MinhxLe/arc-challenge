from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# mirroring, reflection, symmetry

# description:
# In the input, you will see a grid with a single colored object surrounded by a black background.
# To create the output, you need to reflect the colored object across both the vertical and horizontal axes,
# effectively creating a mirrored version of the object. The output grid should show the original object
# in the top-left corner and its mirrored versions in the other quadrants.

def transform(input_grid):
    # Get the bounding box of the colored object
    x, y, width, height = bounding_box(input_grid, background=Color.BLACK)
    
    # Extract the object from the input grid
    object_region = input_grid[x:x+width, y:y+height]
    
    # Create an output grid with a black background
    output_grid = np.full(input_grid.shape, Color.BLACK)
    
    # Place the original object in the top-left corner
    blit_sprite(output_grid, object_region, x=0, y=0, background=Color.BLACK)
    
    # Create the mirrored versions in the other quadrants
    # Vertical mirror
    blit_sprite(output_grid, object_region[:, ::-1], x=0, y=height)
    # Horizontal mirror
    blit_sprite(output_grid, object_region[::-1, :], x=width, y=0, background=Color.BLACK)
    # Diagonal mirror (top-right)
    blit_sprite(output_grid, object_region[::-1, ::-1], x=width, y=height, background=Color.BLACK)

    return output_grid