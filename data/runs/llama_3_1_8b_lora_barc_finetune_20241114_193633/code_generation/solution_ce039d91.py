from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color transformation, symmetry detection, pixel manipulation

# description:
# In the input, you will see a grid with a colored shape and a black background.
# To make the output, create a symmetrical version of the shape along the vertical axis.
# The color of the symmetrical shape should be different from the original shape.

def transform(input_grid):
    # Find the bounding box of the colored shape
    x, y, width, height = bounding_box(input_grid, background=Color.BLACK)
    
    # Extract the shape from the grid
    shape = input_grid[x:x+height, y:y+width]
    
    # Create the output grid initialized to black
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Create a symmetrical version of the shape along the vertical axis
    mirrored_shape = shape[:, ::-1]

    # Change the color of the mirrored shape to a different color (for example, Color.BLUE)
    mirrored_shape[mirrored_shape!= Color.BLACK] = Color.BLUE

    # Place the original shape and the mirrored shape in the output grid
    output_grid[x:x+height, y:y+width] = shape
    output_grid[x:x+height, y+width:y+2*width] = mirrored_shape

    return output_grid