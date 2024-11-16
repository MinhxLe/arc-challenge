from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry, rotation, color mapping

# description:
# In the input, you will see a grid with a colored shape that is not symmetric. 
# To create the output, rotate the shape 90 degrees clockwise and fill the newly exposed areas with a new color. 
# The original shape will be colored in one color, and the new exposed areas will be colored with another color.

def transform(input_grid):
    # Get the bounding box of the non-background pixels
    x, y, width, height = bounding_box(input_grid, background=Color.BLACK)
    
    # Extract the shape from the input grid
    shape = crop(input_grid, background=Color.BLACK)

    # Determine the new color for the exposed areas
    new_color = Color.BLUE  # You can choose any color for the new exposed areas

    # Create an output grid filled with the background color
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Rotate the shape 90 degrees clockwise
    rotated_shape = np.rot90(shape, -1)

    # Blit the original shape in its new position
    blit_sprite(output_grid, shape, x, y, background=Color.BLACK)

    # Fill the newly exposed areas with the new color
    for i in range(rotated_shape.shape[0]):
        for j in range(rotated_shape.shape[1]):
            if rotated_shape[i, j]!= Color.BLACK:  # Only fill if it's not part of the original shape
                output_grid[i + y, j + x] = new_color

    return output_grid