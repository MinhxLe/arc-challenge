from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object movement, color blending, pixel manipulation

# description:
# In the input, you will see a colored object on a black background, and a colored pixel somewhere else on the grid.
# To create the output, move the colored pixel towards the object and blend the colors of the pixel and the object together 
# wherever they overlap, creating a gradient effect.

def transform(input_grid):
    # Create an output grid based on the input grid
    output_grid = np.copy(input_grid)
    
    # Find the coordinates of the colored pixel and the object
    pixel_coords = np.argwhere(output_grid!= Color.BLACK)
    
    if len(pixel_coords) == 0 or len(pixel_coords) == 0:
        return output_grid  # If no colored pixel or object, return original grid
    
    pixel_x, pixel_y = pixel_coords[0]
    object_coords = np.argwhere(output_grid!= Color.BLACK)
    
    # Move the pixel towards the object
    for _ in range(10):  # Move 10 times or until we hit the object
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-way movement
            new_pixel_x, new_pixel_y = pixel_x + dx, pixel_y + dy
            
            if 0 <= new_pixel_x < output_grid.shape[0] and 0 <= new_pixel_y < output_grid.shape[1]:
                if output_grid[new_pixel_x, new_pixel_y] == Color.BLACK:  # Only move to black space
                    output_grid[new_pixel_x, new_pixel_y] = output_grid[pixel_x, pixel_y]  # Move the pixel
                    output_grid[pixel_x, pixel_y] = Color.BLACK  # Remove the original pixel
                    pixel_x, pixel_y = new_pixel_x, new_pixel_y  # Update pixel's position
                    break
                else:
                    # Blend the colors where they overlap
                    output_grid[new_pixel_x, new_pixel_y] = blend_colors(output_grid[new_pixel_x, new_pixel_y], output_grid[pixel_x, pixel_y])
                    break

    return output_grid

def blend_colors(color1, color2):
    # Simple average blend function
    if color1 == Color.BLACK:
        return color2
    if color2 == Color.BLACK:
        return color1
    
    return Color.RED  # Placeholder for blending logic