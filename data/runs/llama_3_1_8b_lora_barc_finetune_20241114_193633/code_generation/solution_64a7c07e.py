from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# mirroring, reflection

# description:
# In the input, you will see a grid with a colored shape that is partially obscured by black pixels. 
# To create the output, mirror the visible part of the shape along the vertical axis and fill in the mirrored section with the same color as the original shape, 
# effectively reconstructing the full shape.

def transform(input_grid):
    # Plan:
    # 1. Extract the visible part of the shape from the input grid.
    # 2. Mirror the visible part along the vertical axis.
    # 3. Fill in the mirrored section in the output grid with the same color as the original shape.

    # Step 1: Find the connected components in the grid, ignoring black pixels
    components = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)

    # Assuming there's only one shape to mirror
    if not components:
        return np.full(input_grid.shape, Color.BLACK)  # If no shape found, return a black grid

    shape = components[0]  # Get the first component (the visible shape)
    # Get the bounding box of the shape
    x, y, w, h = bounding_box(shape)

    # Step 2: Create a mirrored version of the shape
    mirrored_shape = shape[:, ::-1]  # Mirror the shape horizontally

    # Step 3: Create the output grid and place the original shape and mirrored shape
    output_grid = np.full(input_grid.shape, Color.BLACK)
    blit_sprite(output_grid, shape, x=x, y=y, background=Color.BLACK)  # Place the original shape
    blit_sprite(output_grid, mirrored_shape, x=x, y=y + w, background=Color.BLACK)  # Place the mirrored shape

    return output_grid