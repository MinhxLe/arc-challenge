from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# merging, color blending, overlapping shapes

# description:
# In the input, you will see two overlapping colored shapes on a black background. 
# To create the output, merge the two shapes by blending their colors where they overlap, 
# resulting in a new color in that region.

def blend_colors(color1, color2):
    # Simple color blending function
    if color1 == Color.BLACK:
        return color2
    if color2 == Color.BLACK:
        return color1
    # Simple average blending (this could be modified for different blending methods)
    return Color.RED if (color1 == Color.RED or color2 == Color.RED) else color1  # Simplified to show only red blending

def transform(input_grid):
    # Detect the two objects in the input grid
    objects = detect_objects(grid=input_grid, monochromatic=False, connectivity=8)
    output_grid = np.zeros_like(input_grid)

    # Iterate through the grid and blend colors where the two objects overlap
    for obj in objects:
        for x in range(obj.shape[0]):
            for y in range(obj.shape[1]):
                if obj[x, y]!= Color.BLACK:  # Only consider non-background pixels
                    # Get the color of the current pixel
                    color1 = obj[x, y]
                    # Check the overlapping area
                    for dx in range(-1, 2):  # Check 3x3 area centered at the current pixel
                        for dy in range(-1, 2):
                            if 0 <= x + dx < output_grid.shape[0] and 0 <= y + dy < output_grid.shape[1]:
                                # Blend colors
                                output_grid[x + dx, y + dy] = blend_colors(output_grid[x + dx, y + dy], color1)

    return output_grid