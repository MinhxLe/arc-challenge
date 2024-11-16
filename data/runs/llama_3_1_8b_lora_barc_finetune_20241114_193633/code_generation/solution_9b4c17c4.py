from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object separation, color blending

# description:
# In the input, you will see two overlapping objects of different colors. 
# To create the output, separate the two objects while blending their colors in the overlapping region to create a new color.

def transform(input_grid):
    # Step 1: Detect the two overlapping objects in the grid
    objects = detect_objects(grid=input_grid, monochromatic=False, connectivity=4, allowed_dimensions=[(1, 1)])

    # We assume there are exactly two objects
    assert len(objects) == 2

    # Step 2: Crop the two objects from the input grid
    object1 = crop(objects[0], background=Color.BLACK)
    object2 = crop(objects[1], background=Color.BLACK)

    # Step 3: Determine the overlap area
    overlap = np.logical_and(object1, object2)

    # Step 4: Blend the colors in the overlap
    # Get the colors of the two objects
    color1 = object1[overlap]
    color2 = object2[overlap]

    # Create a new color in the overlapping region by blending the two colors
    blended_color = np.where(color1 == Color.RED, Color.RED, Color.BLUE)  # Placeholder blending logic

    # Step 5: Create the output grid
    output_grid = np.full(input_grid.shape, Color.BLACK)
    blit_sprite(output_grid, object1, x=0, y=0, background=Color.BLACK)
    blit_sprite(output_grid, object2, x=0, y=0, background=Color.BLACK)

    # Step 6: Blit the blended color in the overlap area
    output_grid[overlap] = blended_color

    return output_grid