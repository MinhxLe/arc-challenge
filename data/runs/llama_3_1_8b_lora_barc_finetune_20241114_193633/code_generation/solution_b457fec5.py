from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color transformation, boundary extraction, gradient fill

# description:
# In the input, you will see a grid containing a colorful object surrounded by a border of black pixels. 
# The object can have various colors, but there is a single pixel in the bottom right corner that indicates 
# the color to fill the entire object with. To make the output, extract the object from the grid, 
# fill it with the color from the bottom right corner, and place it back into a new grid with a black background.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Step 1: Detect the colorful object surrounded by black pixels
    colorful_objects = detect_objects(grid=input_grid, colors=Color.NOT_BLACK, monochromatic=False, connectivity=4)

    # Step 2: Assume there's only one object, extract it
    if len(colorful_objects)!= 1:
        raise ValueError("Expected exactly one colorful object surrounded by black pixels.")
    colorful_object = colorful_objects[0]

    # Step 3: Get the color from the bottom right corner
    fill_color = input_grid[-1, -1]

    # Step 4: Crop the colorful object
    object_sprite = crop(colorful_object, background=Color.BLACK)

    # Step 5: Create a new output grid filled with black
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Step 6: Fill the cropped object with the extracted color
    output_grid = blit_sprite(output_grid, object_sprite, x=1, y=1, background=Color.BLACK)
    output_grid[object_sprite!= Color.BLACK] = fill_color

    return output_grid