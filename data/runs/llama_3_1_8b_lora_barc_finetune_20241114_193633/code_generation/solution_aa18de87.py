from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, color filling, grid transformation

# description:
# In the input, you will see a grid with a colored shape that is symmetric along the vertical axis.
# To make the output, fill the shape with a contrasting color on the opposite side to enhance its symmetry.

def transform(input_grid):
    # Find the bounding box of the colored shape
    x, y, width, height = bounding_box(input_grid, background=Color.BLACK)
    
    # Extract the shape from the input grid
    shape = input_grid[x:x + height, y:y + width]
    
    # Create a contrasting color for filling
    contrasting_color = Color.RED if np.any(shape!= Color.BLACK) else Color.BLACK
    
    # Create the output grid initialized to black
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Fill the shape with the contrasting color on the opposite side
    mirrored_shape = np.copy(shape[:, ::-1])  # Mirror the shape horizontally
    output_grid[x:x + height, y:y + width] = shape
    output_grid[x:x + height, y + width:y + 2 * width] = mirrored_shape

    return output_grid