from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, overlapping shapes

# description:
# In the input, you will see two overlapping shapes, each with a distinct color. 
# To create the output, blend the two colors together in the overlapping region to create a new color that represents the overlap.
# The output should be a grid that represents the two shapes with their respective colors and the blended region in the middle.

def blend_colors(color1, color2):
    # Simple color blending function
    # If colors are the same, return the first color
    if color1 == color2:
        return color1
    # For simplicity, let's create a basic blend of two colors
    # In a more advanced scenario, you could define your own blending rules
    return Color.BLUE if (color1 == Color.RED and color2 == Color.GREEN) else Color.YELLOW

def transform(input_grid):
    # Detect the two shapes in the input grid
    objects = detect_objects(grid=input_grid, monochromatic=False, connectivity=8, background=Color.BLACK)
    
    # Create an output grid initialized to the background color
    output_grid = np.full_like(input_grid, Color.BLACK)
    
    # Iterate through each object to blend colors
    for obj in objects:
        # Get the bounding box of the current object
        x, y, width, height = bounding_box(obj, background=Color.BLACK)
        sprite = crop(obj, background=Color.BLACK)
        
        # Place the original sprite in the output grid
        blit_sprite(output_grid, sprite, x, y, background=Color.BLACK)
        
        # Blend the colors in the overlapping region
        for i in range(width):
            for j in range(height):
                if sprite[i, j]!= Color.BLACK:  # Check if it's not a background pixel
                    # Get the color of the current pixel
                    current_color = sprite[i, j]
                    # Check if this pixel overlaps with another object
                    for other_obj in objects:
                        if np.any(other_obj!= Color.BLACK):
                            # Blend with the other color
                            blended_color = blend_colors(current_color, other_obj[0, 0])  # Assume other_obj is a single color
                            if blended_color!= Color.BLACK:  # Only place if not background
                                output_grid[y + i, x + j] = blended_color
                            else:
                                output_grid[y + i, x + j] = current_color  # Keep original color if blending fails
                        else:
                            output_grid[y + i, x + j] = current_color  # Keep original color if no overlap

    return output_grid