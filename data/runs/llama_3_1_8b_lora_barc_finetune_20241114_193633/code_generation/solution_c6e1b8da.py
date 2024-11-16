from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# layering, color blending, transparency

# description:
# In the input, you will see several layers of colored objects stacked on top of each other on a black background.
# To create the output, you should blend the colors of the layers where they overlap, resulting in new colors where the overlapping regions are located.

def transform(input_grid):
    # Plan:
    # 1. Identify the layers in the input grid
    # 2. For each layer, blend it with the existing grid where they overlap

    output_grid = np.copy(input_grid)
    
    # Find connected components in the grid (layers)
    layers = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)
    
    for layer in layers:
        # Get the bounding box of the layer
        x, y, width, height = bounding_box(layer)
        
        # Get the color of the current layer
        layer_color = layer[x, y]
        
        # Blend the layer color with the existing output grid where they overlap
        for i in range(height):
            for j in range(width):
                if layer[i, j]!= Color.BLACK:
                    # Blend the colors if the pixel is not black in the output grid
                    if output_grid[x + i, y + j]!= Color.BLACK:
                        # Simple blending rule: if both colors are not black, average them
                        blended_color = blend_colors(output_grid[x + i, y + j], layer_color)
                        output_grid[x + i, y + j] = blended_color
                    else:
                        # If the output pixel is black, just set it to the layer color
                        output_grid[x + i, y + j] = layer_color
    
    return output_grid

def blend_colors(color1, color2):
    # Simple blending function that returns a new color based on the average of the two colors
    # Here we assume colors are represented as integers, and we will average their indices
    if color1 == Color.BLACK:
        return color2
    elif color2 == Color.BLACK:
        return color1
    else:
        # Average the color indices
        # Assuming Color is represented as integers, we can take the average as follows
        return Color(int((color1 + color2) // 2))