from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color merging, object transformation

# description:
# In the input, you will see a grid containing various colored objects. 
# To create the output, you will merge all objects of the same color into one larger object 
# while maintaining their positions relative to each other. The output should be a grid 
# with a single object for each unique color in the input grid.

def transform(input_grid):
    # Create an output grid initialized to the background color
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Find all connected components in the input grid
    components = find_connected_components(input_grid, monochromatic=True, background=Color.BLACK, connectivity=4)

    # For each component, find its bounding box and color
    for component in components:
        # Get the bounding box of the component
        x, y, w, h = bounding_box(component)
        
        # Crop the component to create the sprite
        sprite = crop(component)

        # Determine the color of the current component
        color = np.unique(sprite[sprite!= Color.BLACK])[0]

        # Determine the position to place the component in the output grid
        output_x = x
        output_y = y
        
        # Check if the position is already occupied by another object
        if output_x < 0 or output_x + w >= output_grid.shape[0] or output_y < 0 or output_y + h >= output_grid.shape[1]:
            continue
        
        # Place the sprite in the output grid
        blit_sprite(output_grid, sprite, x=output_x, y=output_y, background=Color.BLACK)

    return output_grid