from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# boundary detection, color transformation, gradient creation

# description:
# In the input, you will see a grid with a colored shape and a black background. 
# The shape has a distinct boundary defined by a different color (e.g., teal). 
# To create the output, fill the area inside the boundary with a gradient that transitions from 
# the original color of the shape to black.

def transform(input_grid):
    # Create an output grid initialized to black
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Find the boundary of the shape
    boundary_mask = object_boundary(input_grid, background=Color.BLACK, connectivity=4)

    # Get the color of the shape
    shape_color = input_grid[boundary_mask][0]

    # Create a gradient fill from the shape color to black
    for x in range(output_grid.shape[0]):
        for y in range(output_grid.shape[1]):
            if boundary_mask[x, y]:
                # Check if the pixel is part of the shape
                output_grid[x, y] = shape_color
            else:
                # Calculate the distance from the boundary
                distance = np.linalg.norm(np.array([x, y]) - np.array(boundary_mask.shape) / 2)
                # Normalize the distance to get a gradient value
                gradient_value = min(1, distance / (boundary_mask.shape[0] // 2))
                # Interpolate color from shape_color to black
                output_grid[x, y] = blend_colors(shape_color, Color.BLACK, gradient_value)

    return output_grid

def blend_colors(color1, color2, blend_factor):
    """
    Blends two colors based on the blend factor (0.0 to 1.0).
    This function assumes colors are represented as strings, not integers.
    """
    # Here we will assume colors are represented as strings
    # Create a mapping from color to RGB values (for simplicity, we'll use dummy RGB values)
    color_map = {
        Color.BLUE: (0, 0, 255),
        Color.RED: (255, 0, 0),
        Color.GREEN: (0, 255, 0),
        Color.YELLOW: (255, 255, 0),
        Color.GRAY: (128, 128, 128),
        Color.PINK: (255, 192, 203),
        Color.ORANGE: (255, 165, 0),
        Color.PURPLE: (0, 128, 128),
        Color.BROWN: (128, 0, 0),
        Color.BLACK: (0, 0, 0)
    }
    
    r1, g1, b1 = color_map[color1]
    r2, g2, b2 = color_map[Color.BLACK]  # black is treated as (0, 0, 0)

    # Calculate blended color
    blended_r = int(r1 * (1 - blend_factor) + r2 * blend_factor)
    blended_g = int(g1 * (1 - blend_factor) + g2 * blend_factor)
    blended_b = int(b1 * (1 - blend_factor) + b2 * blend_factor)

    # Convert back to a color name (this could be improved to use a more sophisticated method)
    blended_color = Color.RED if blended_r == 255 and blended_g == 0 and blended_b == 0 else Color.BLACK
    return blended_color