from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# merging, color blending

# description:
# In the input, you will see two overlapping colored shapes on a black background. 
# To make the output, merge the two shapes by blending their colors where they overlap,
# creating a new color in the overlapping region. The output should retain the original shapes' 
# non-overlapping regions.

def transform(input_grid):
    # Create an output grid initialized to black
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Find the connected components in the input grid
    objects = find_connected_components(input_grid, monochromatic=False, connectivity=4)

    # Merge the colors of the overlapping regions
    for obj in objects:
        # Get the bounding box of the object
        x, y, w, h = bounding_box(obj, background=Color.BLACK)
        # Crop the object to isolate it
        sprite = crop(obj, background=Color.BLACK)
        
        # Determine the color of the object
        color = np.unique(sprite[sprite!= Color.BLACK])[0]
        
        # Blit the object to the output grid
        blit_sprite(output_grid, sprite, x=x, y=y, background=Color.BLACK)

    # Now blend colors in overlapping regions
    for i in range(output_grid.shape[0]):
        for j in range(output_grid.shape[1]):
            if output_grid[i, j]!= Color.BLACK:
                # Check if the pixel is part of the merged region
                if (i > 0 and output_grid[i-1, j]!= Color.BLACK) or (i < output_grid.shape[0] - 1 and output_grid[i+1, j]!= Color.BLACK) or
                   (j > 0 and output_grid[i, j-1]!= Color.BLACK) or (j < output_grid.shape[1] - 1 and output_grid[i, j+1]!= Color.BLACK):
                    # Blend the colors if there's overlap
                    if output_grid[i, j]!= Color.BLACK:
                        # Here we can define a simple blending rule, for example, mixing colors
                        output_grid[i, j] = blend_colors(output_grid[i, j], Color.GRAY)  # Example blending to gray

    return output_grid

def blend_colors(color1, color2):
    """
    A simple blending function that returns a new color based on the two colors.
    In this case, we will blend by averaging the color values, treating them as RGB-like.
    Here we will just return gray as a blend for simplicity.
    """
    return Color.GRAY