from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, symmetry, mirroring, reflection

# description:
# In the input, you will see a grid containing a colored shape on a black background. 
# The output should be a mirrored version of the shape across the vertical centerline of the grid, 
# with the mirrored shape colored in a distinct color (e.g., blue).

def transform(input_grid):
    # Find the shape in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=False)
    
    # Assuming there's only one shape in the input
    shape = objects[0]

    # Get the bounding box of the shape
    x, y, width, height = bounding_box(shape)

    # Create the output grid, initially filled with the background color
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Calculate the centerline for mirroring
    center_x = input_grid.shape[1] // 2

    # Fill the output grid with the original shape
    blit_sprite(output_grid, shape, x=x, y=y, background=Color.BLACK)

    # Mirror the shape across the vertical centerline
    mirrored_shape = shape[:, ::-1]  # Mirror the shape horizontally
    mirrored_shape[mirrored_shape!= Color.BLACK] = Color.BLUE  # Change the color to blue

    # Calculate the position to blit the mirrored shape
    mirrored_x = x + (center_x - (x + width) // 2)  # Adjust to center the mirrored shape

    # Blit the mirrored shape onto the output grid
    blit_sprite(output_grid, mirrored_shape, x=mirrored_x, y=y, background=Color.BLACK)

    return output_grid