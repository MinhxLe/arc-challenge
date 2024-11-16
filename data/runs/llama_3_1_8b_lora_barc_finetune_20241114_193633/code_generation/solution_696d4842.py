from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, transparency

# description:
# In the input, you will see two overlapping colored objects on a black background. 
# To make the output, create a new color that is a blend of the two colors, where the overlap is indicated by the new color, while the non-overlapping parts remain unchanged.

def blend_colors(color1, color2):
    """
    Simple function to blend two colors. The blending is done by averaging their RGB values,
    assuming colors are represented as integers. Here, we will simplify and just return 
    a new color based on their indices (as a proxy for blending).
    """
    # Simple blending logic: average the two colors
    if color1 == Color.BLACK:
        return color2
    if color2 == Color.BLACK:
        return color1
    # For simplicity, we can use a simple mapping of indices to blended colors
    color_map = {
        Color.BLUE: 1,
        Color.RED: 2,
        Color.GREEN: 3,
        Color.YELLOW: 4,
        Color.GRAY: 5,
        Color.PINK: 6,
        Color.ORANGE: 7,
        Color.PURPLE: 8,
        Color.BROWN: 9,
        Color.BLACK: 0
    }
    return Color.BLACK if color_map[color1] + color_map[color2] >= len(Color.ALL_COLORS) else Color.ALL_COLORS[color_map[color1] + color_map[color2]]

def transform(input_grid):
    # Copy the input grid to the output grid
    output_grid = np.copy(input_grid)
    
    # Detect connected components in the input grid
    objects = detect_objects(input_grid, background=Color.BLACK, monochromatic=True, connectivity=4)
    
    # For each connected component, blend colors where they overlap
    for obj in objects:
        # Get the bounding box of the object
        x, y, w, h = bounding_box(obj)
        # Create a mask for the object
        mask = obj!= Color.BLACK
        
        # Iterate over the pixels in the bounding box
        for i in range(h):
            for j in range(w):
                if mask[i, j]:  # Only process the object's pixels
                    # Check for overlap with another object
                    overlap_color = Color.BLACK
                    for other_obj in objects:
                        if np.any(other_obj!= Color.BLACK) and other_obj[i, j]!= Color.BLACK:
                            overlap_color = other_obj[i, j]
                            break
                    
                    # Blend colors if there's an overlap
                    if overlap_color!= Color.BLACK:
                        output_grid[y + i, x + j] = overlap_color
    
    return output_grid