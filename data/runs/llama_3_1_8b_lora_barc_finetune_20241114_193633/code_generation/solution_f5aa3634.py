from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# shape detection, color grouping, spatial arrangement

# description:
# In the input, you will see various colored shapes arranged on a black background. The shapes can overlap.
# To create the output grid, you should identify distinct shapes, group them by color, and arrange them in a grid format.
# Each color group should be placed in a separate section, arranged in the order of their appearance from top to bottom.

def transform(input_grid):
    # Find all the connected components (shapes) in the input grid
    shapes = find_connected_components(input_grid, background=Color.BLACK, monochromatic=False, connectivity=4)
    
    # Prepare an output grid to hold the arranged shapes
    output_height = len(shapes) * 3  # Assuming each shape will be 3x3
    output_grid = np.full((output_height, 3), Color.BLACK, dtype=int)

    # Define a color mapping for shapes to their corresponding color groups
    color_groups = {}
    for shape in shapes:
        # Get the color of the shape
        color = shape[0, 0]  # Assuming monochromatic within the shape
        if color not in color_groups:
            color_groups[color] = []
        color_groups[color].append(crop(shape, background=Color.BLACK))
    
    # Arrange shapes in the output grid
    current_y = 0
    for color, shapes in color_groups.items():
        for shape in shapes:
            # Place the shape in the output grid
            blit_sprite(output_grid, shape, x=0, y=current_y, background=Color.BLACK)
            current_y += 3  # Move down for the next shape
    
    return output_grid