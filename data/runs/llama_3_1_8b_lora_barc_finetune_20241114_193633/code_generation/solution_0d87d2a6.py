from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object detection, color mapping, shape transformation

# description:
# In the input, you will see a grid filled with various colored shapes. 
# The goal is to create an output grid where each colored shape is transformed:
# 1. Each shape's color is changed based on its position in the grid:
#    - Shapes in the top half of the grid will be colored blue.
#    - Shapes in the bottom half of the grid will be colored red.
# 2. If a shape touches the border of the grid, it will be colored green.
# 3. The shapes should be centered in the output grid while maintaining their original size and shape.

def transform(input_grid):
    # Find all connected components in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True, connectivity=8)
    
    # Prepare the output grid
    output_grid = np.full(input_grid.shape, Color.BLACK, dtype=int)

    for obj in objects:
        # Get the bounding box of the current shape
        x, y, w, h = bounding_box(obj, background=Color.BLACK)

        # Crop the shape to get the actual sprite
        sprite = crop(obj, background=Color.BLACK)

        # Determine the color based on the position of the shape
        if y < output_grid.shape[0] / 2:
            # Top half
            color = Color.BLUE
        else:
            # Bottom half
            color = Color.RED
        
        # Check if the shape touches the border
        if np.any(obj[0, :]!= Color.BLACK) or np.any(obj[-1, :]!= Color.BLACK) or np.any(obj[:, 0]!= Color.BLACK) or np.any(obj[:, -1]!= Color.BLACK):
            color = Color.GREEN
        
        # Blit the transformed shape to the output grid, centered
        blit_sprite(output_grid, sprite, x=(output_grid.shape[0] // 2) - (sprite.shape[0] // 2), y=(output_grid.shape[1] // 2) - (sprite.shape[1] // 2), background=Color.BLACK)
        
        # Color the shape based on the determined color
        output_grid[(output_grid == Color.BLUE) & (sprite!= Color.BLACK)] = color
        output_grid[(output_grid == Color.RED) & (sprite!= Color.BLACK)] = color
        output_grid[(output_grid == Color.GREEN) & (sprite!= Color.BLACK)] = color

    return output_grid