from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# rotation, color mapping

# description:
# In the input, you will see a colored object in the center of a grid surrounded by a black background.
# To make the output, rotate the object by 90 degrees clockwise and change its color to the color of the pixel in the center of the grid.

def transform(input_grid):
    # Find the center of the grid
    center_x, center_y = input_grid.shape[0] // 2, input_grid.shape[1] // 2
    
    # Crop the object from the grid
    object_sprite = crop(input_grid, background=Color.BLACK)
    
    # Get the color of the center pixel
    center_color = input_grid[center_x, center_y]
    
    # Rotate the object 90 degrees clockwise
    rotated_object = np.rot90(object_sprite, k=-1)
    
    # Create the output grid and fill it with the center color
    output_grid = np.full_like(input_grid, center_color)
    
    # Calculate the position to place the rotated object in the output grid
    obj_height, obj_width = rotated_object.shape
    output_x = center_x - obj_height // 2
    output_y = center_y - obj_width // 2
    
    # Blit the rotated object onto the output grid
    blit_sprite(output_grid, rotated_object, x=output_x, y=output_y, background=center_color)
    
    return output_grid