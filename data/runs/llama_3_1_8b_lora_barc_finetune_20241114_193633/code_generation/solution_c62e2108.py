from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# boundary detection, color filling, transformation

# description:
# In the input, you will see a grid with several colored shapes, some of which are touching the boundary of the grid.
# To create the output, identify all shapes that are touching the boundary and fill them with a specific color (e.g., blue),
# while leaving the inner shapes unchanged.

def transform(input_grid):
    # Create an output grid initialized to black
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Find all connected components in the input grid
    components = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)
    
    # Identify shapes that touch the boundary
    boundary_shapes = []
    for component in components:
        # Get the bounding box of the component
        x, y, width, height = bounding_box(component, background=Color.BLACK)
        # Check if the shape touches the boundary of the grid
        if x == 0 or y == 0 or x + width - 1 >= input_grid.shape[0] - 1 or y + height - 1 >= input_grid.shape[1] - 1:
            boundary_shapes.append(component)
    
    # Fill the boundary shapes with blue color
    for shape in boundary_shapes:
        # Crop the shape to get the actual sprite
        sprite = crop(shape, background=Color.BLACK)
        # Determine the top-left position to place the sprite in the output grid
        x, y = np.argwhere(shape!= Color.BLACK)[0]  # Assume there's only one component touching the boundary
        # Blit the blue sprite onto the output grid
        output_grid = blit_sprite(output_grid, sprite, x, y, background=Color.BLACK)

    return output_grid