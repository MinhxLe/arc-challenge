from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# shape transformation, reflection, color preservation

# description:
# In the input, you will see a colored shape on a black background. 
# To make the output, create a mirror image of the shape on the opposite side of the grid 
# while maintaining the same color. The output grid should be the same size as the input grid.

def transform(input_grid):
    # Create an output grid that is a copy of the input grid
    output_grid = np.copy(input_grid)

    # Find the bounding box of the colored shape
    x, y, width, height = bounding_box(input_grid!= Color.BLACK)

    # Extract the shape from the input grid
    shape = input_grid[x:x+width, y:y+height]

    # Create the mirror image of the shape
    mirrored_shape = shape[:, ::-1]

    # Blit the mirrored shape onto the output grid
    output_grid[x:x+width, y+height:] = mirrored_shape

    return output_grid