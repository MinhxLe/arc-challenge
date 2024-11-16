from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color counting, object detection, grid manipulation

# description:
# In the input, you will see several objects of different colors scattered across a grid. 
# Each object consists of a single color and can be of varying sizes. The output should be a 
# new grid where each object is replaced with a new color based on the number of pixels it contains. 
# The new color will be determined by counting the number of pixels in each object:
# - If the object has 1 pixel, color it blue.
# - If it has 2 pixels, color it green.
# - If it has 3 pixels, color it red.
# - If it has 4 pixels, color it yellow.
# - If it has more than 4 pixels, color it orange.

def transform(input_grid):
    # Plan:
    # 1. Detect all objects in the grid
    # 2. Count the number of pixels in each object
    # 3. Create a new grid and color each object based on its pixel count

    # Detect all objects in the input grid
    objects = detect_objects(input_grid, monochromatic=False, connectivity=4)

    # Initialize the output grid with the same size as the input grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    for obj in objects:
        # Count the number of pixels in the object
        pixel_count = np.sum(obj!= Color.BLACK)

        # Determine the new color based on the pixel count
        if pixel_count == 1:
            new_color = Color.BLUE
        elif pixel_count == 2:
            new_color = Color.GREEN
        elif pixel_count == 3:
            new_color = Color.RED
        elif pixel_count == 4:
            new_color = Color.YELLOW
        else:
            new_color = Color.ORANGE
        
        # Get the position of the object in the original grid
        obj_x, obj_y = object_position(obj, background=Color.BLACK, anchor="upper left")
        
        # Blit the new colored object onto the output grid
        output_grid = blit_sprite(output_grid, np.full(obj.shape, new_color), x=obj_x, y=obj_y, background=Color.BLACK)

    return output_grid