from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, overlapping patterns

# description:
# In the input, you will see two overlapping patterns made of colored pixels on a black background.
# To create the output, blend the colors of the overlapping pixels using a simple averaging method to create a new color.
# The output grid should reflect the new blended colors wherever the two patterns overlap, while preserving the original colors in non-overlapping areas.

def blend_colors(color1, color2):
    """
    Simple average function for blending two colors.
    This is a placeholder function and should be adjusted for actual color blending.
    """
    # Define a mapping of colors to RGB values for blending
    color_map = {
        Color.RED: (255, 0, 0),
        Color.GREEN: (0, 255, 0),
        Color.BLUE: (0, 0, 255),
        Color.YELLOW: (255, 255, 0),
        Color.ORANGE: (255, 165, 0),
        Color.PINK: (255, 192, 203),
        Color.PURPLE: (0, 128, 128),
        Color.BROWN: (128, 0, 0),
        Color.GRAY: (128, 128, 128),
        Color.BLACK: (0, 0, 0),
    }
    
    # Get RGB values
    rgb1 = np.array(color_map[color1])
    rgb2 = np.array(color_map[color2])
    
    # Blend by averaging
    blended_rgb = (rgb1 + rgb2) / 2
    # Find the closest color in the color_map
    blended_color = min(color_map.keys(), key=lambda c: np.linalg.norm(color_map[c] - blended_rgb))
    return blended_color

def transform(input_grid):
    # Create the output grid initialized to black
    output_grid = np.full(input_grid.shape, Color.BLACK)
    
    # Detect connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4)
    
    for obj in objects:
        # Get the bounding box of the object
        x, y, w, h = bounding_box(obj, background=Color.BLACK)
        
        # Extract the sprite
        sprite = crop(obj, background=Color.BLACK)

        # Blend the sprite with the output grid
        for i in range(sprite.shape[0]):
            for j in range(sprite.shape[1]):
                if sprite[i, j]!= Color.BLACK:  # Only blend if it's not black
                    # Check for overlap with the output grid
                    if output_grid[y + i, x + j]!= Color.BLACK:
                        # Blend the colors
                        output_grid[y + i, x + j] = blend_colors(sprite[i, j], output_grid[y + i, x + j])
                    else:
                        # Otherwise, just place the sprite color
                        output_grid[y + i, x + j] = sprite[i, j]

    return output_grid