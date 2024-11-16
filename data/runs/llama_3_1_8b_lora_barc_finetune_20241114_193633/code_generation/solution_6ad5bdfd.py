from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object stacking, gravity, alignment

# description:
# In the input, you will see several colored objects stacked in a grid, each occupying a rectangular area. The background is black. 
# To create the output grid, simulate gravity by moving each object downwards until it touches the bottom of the grid or another object. 
# The output should display the new positions of all objects, aligned at the bottom of the grid.

def transform(input_grid):
    # Plan:
    # 1. Find the objects in the input grid
    # 2. For each object, calculate its new position based on gravity
    # 3. Blit each object onto the output grid in its new position

    # Step 1: Find all connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)

    # Create an output grid of the same width but with a height equal to the maximum height of the objects
    output_grid = np.full((input_grid.shape[0], input_grid.shape[1]), Color.BLACK)

    # Step 2: For each object, determine its new position
    for obj in objects:
        # Get the bounding box of the object
        x, y, width, height = bounding_box(obj, background=Color.BLACK)

        # Calculate the new position based on gravity
        new_y = output_grid.shape[0] - height  # New position at the bottom
        # Center the object within the output grid
        new_x = (output_grid.shape[1] - width) // 2

        # Place the object in the output grid
        blit_sprite(output_grid, obj, x=new_x, y=new_y, background=Color.BLACK)

    return output_grid