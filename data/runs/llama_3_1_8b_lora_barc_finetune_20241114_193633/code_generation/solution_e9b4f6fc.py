from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object detection, shape transformation

# description:
# In the input, you will see a grid with various colored objects and a background of black pixels. 
# To create the output, you need to find all objects of a specific color (say, red) and replace them with a new shape 
# based on their size (e.g., a square or circle). The output should maintain the original color of the objects.

def transform(input_grid):
    # Step 1: Detect all red objects in the grid
    red_objects = detect_objects(grid=input_grid, colors=[Color.RED], monochromatic=True, connectivity=4, allowed_dimensions=None, colors=None, can_overlap=False)
    
    # Step 2: Create an output grid initialized to black
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Step 3: For each detected red object, replace it with a shape
    for obj in red_objects:
        # Get the bounding box of the red object
        x, y, w, h = bounding_box(obj)
        sprite = crop(obj)
        
        # Determine the shape to use based on the size of the object
        if w == h:  # Square
            shape = np.full((w, h), Color.RED)  # Maintain original color
        else:  # Rectangle
            shape = np.full((h, w), Color.RED)  # Maintain original color
        
        # Blit the shape onto the output grid at the corresponding position
        blit_sprite(output_grid, shape, x=x, y=y, background=Color.BLACK)
    
    return output_grid