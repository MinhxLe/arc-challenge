from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry, reflection, color mapping

# description:
# In the input, you will see a grid containing a colored shape and a single colored pixel at a random location.
# To make the output, you should reflect the shape across the center of the grid and fill the new area with the color of the pixel you removed.

def transform(input_grid):
    # Step 1: Detect the color of the pixel that will be removed (the colored pixel)
    colored_pixel_objects = detect_objects(grid=input_grid, colors=Color.NOT_BLACK, monochromatic=True, connectivity=4, allowed_dimensions=[(1, 1)])
    assert len(colored_pixel_objects) == 1  # There should be exactly one colored pixel
    colored_pixel_object = colored_pixel_objects[0]
    color_to_fill = colored_pixel_object[0, 0]

    # Step 2: Get the position of the colored pixel
    colored_pixel_position = np.argwhere(colored_pixel_object!= Color.BLACK)
    colored_x, colored_y = colored_pixel_position[0]

    # Step 3: Create the output grid initialized to black
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Step 4: Reflect the shape across the center of the grid
    reflected_shape = np.copy(input_grid)
    center_x = input_grid.shape[0] // 2
    center_y = input_grid.shape[1] // 2
    reflected_shape = np.flipud(np.fliplr(reflected_shape))

    # Step 5: Fill the new area with the color of the removed pixel
    output_grid[reflected_shape!= Color.BLACK] = color_to_fill

    # Step 6: Blit the reflected shape onto the output grid
    output_grid = blit_sprite(grid=output_grid, sprite=reflected_shape, x=0, y=0, background=Color.BLACK)

    return output_grid