from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object extraction, color mapping

# description:
# In the input, you will see two distinct objects, one in red and one in blue, on a black background.
# To make the output, extract each object, change their colors to green and pink respectively, and place them 
# into a new grid with a black background.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a blank new canvas for the output
    output_grid = np.zeros_like(input_grid)

    # Detect the objects in the input grid
    red_objects = detect_objects(grid=input_grid, colors=[Color.RED], monochromatic=True, connectivity=4, allowed_dimensions=[(3, 3)])
    blue_objects = detect_objects(grid=input_grid, colors=[Color.BLUE], monochromatic=True, connectivity=4, allowed_dimensions=[(3, 3)])

    # We expect exactly one red and one blue object
    assert len(red_objects) == 1 and len(blue_objects) == 1
    red_object = red_objects[0]
    blue_object = blue_objects[0]

    # Crop the red and blue objects to get rid of excess black pixels
    red_sprite = crop(red_object, background=Color.BLACK)
    blue_sprite = crop(blue_object, background=Color.BLACK)

    # Change colors
    red_sprite[red_sprite!= Color.BLACK] = Color.GREEN
    blue_sprite[blue_sprite!= Color.BLACK] = Color.PINK

    # Blit the colored sprites into the output grid
    output_grid = blit_sprite(output_grid, red_sprite, x=1, y=1, background=Color.BLACK)  # Place red sprite
    output_grid = blit_sprite(output_grid, blue_sprite, x=1, y=3, background=Color.BLACK)  # Place blue sprite

    return output_grid