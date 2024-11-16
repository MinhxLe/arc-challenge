from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, object alignment

# description:
# In the input, you will see a grid with a black background and several colored objects (represented in different colors).
# To make the output, align each colored object to a specific position in the grid based on its color, 
# while ensuring that they do not overlap. The alignment positions are determined by their colors: 
# red objects align to (1,1), blue to (1,3), green to (3,1), and yellow to (3,3).
# The output grid should be filled with the colors in the specified positions.

def transform(input_grid):
    # Initialize the output grid with a black background
    output_grid = np.full(input_grid.shape, Color.BLACK, dtype=int)

    # Define the alignment positions based on colors
    alignment_positions = {
        Color.RED: (1, 1),
        Color.BLUE: (1, 3),
        Color.GREEN: (3, 1),
        Color.YELLOW: (3, 3)
    }

    # Detect the objects in the input grid
    objects = detect_objects(grid=input_grid, colors=Color.NOT_BLACK, monochromatic=True, connectivity=4)

    # Place each object in the output grid at its corresponding alignment position
    for obj in objects:
        # Crop the object to get its bounding box
        cropped_obj = crop(obj, background=Color.BLACK)
        
        # Get the color of the object
        color = cropped_obj[0, 0]
        
        # Determine the target position based on the color
        target_x, target_y = alignment_positions[color]
        
        # Get the position of the object in the input grid
        obj_x, obj_y = object_position(obj, background=Color.BLACK, anchor="upper left")
        
        # Place the cropped object in the output grid at the target position
        blit_sprite(output_grid, cropped_obj, target_x, target_y, background=Color.BLACK)
    
    return output_grid