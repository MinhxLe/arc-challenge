from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, transparency, layering

# description:
# In the input, you will see a grid with a colored pattern on a black background. 
# To create the output, overlay a semi-transparent layer of the same color pattern over the original,
# allowing the colors to blend where they overlap. The transparency should be represented by blending
# the colors in the overlapping regions based on a simple rule: if the pixel in the output grid is already
# a color, the new color will blend with it based on a simple rule: if the original pixel is black, it takes the new color,
# otherwise, it remains unchanged.

def transform(input_grid):
    # Create a copy of the input grid to serve as the output grid
    output_grid = np.copy(input_grid)
    
    # Find the connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=8)
    
    # For each connected component, blend the colors
    for obj in objects:
        # Get the bounding box of the object
        x, y, width, height = bounding_box(obj)
        
        # Create a mask for the object
        mask = (obj!= Color.BLACK)
        
        # Iterate through each pixel in the object
        for i in range(obj.shape[0]):
            for j in range(obj.shape[1]):
                if mask[i, j]:
                    # Get the color of the pixel
                    original_color = obj[i, j]
                    
                    # Get the position in the output grid
                    output_x, output_y = x + i, y + j
                    
                    # Check if the pixel in the output grid is black (transparent)
                    if output_grid[output_x, output_y] == Color.BLACK:
                        # If it's black, we can set it to the original color
                        output_grid[output_x, output_y] = original_color
                    else:
                        # Blend the colors
                        # Simple average blending
                        output_grid[output_x, output_y] = blend_colors(output_grid[output_x, output_y], original_color)

    return output_grid

def blend_colors(color1, color2):
    # Simple blending function: average the color values
    if color1 == Color.BLACK:
        return color2
    if color2 == Color.BLACK:
        return color1
    return color1  # In this case, we're keeping the first color for simplicity