from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, radial symmetry

# description:
# In the input, you will see a grid with a central colored pixel surrounded by a ring of pixels of various colors.
# To create the output, blend the colors of the ring of pixels with the color of the central pixel, creating a gradient effect that radiates outward from the center.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)
    
    # Get the dimensions of the input grid
    height, width = input_grid.shape

    # Find the central pixel color
    center_color = input_grid[height // 2, width // 2]

    # Find the surrounding pixels
    surrounding_pixels = []
    for x in range(height):
        for y in range(width):
            if (x == 0 or x == height - 1 or y == 0 or y == width - 1) and input_grid[x, y]!= Color.BLACK:
                surrounding_pixels.append((x, y, input_grid[x, y]))

    # Blend colors and fill the output grid
    for (x, y, color) in surrounding_pixels:
        distance = max(abs(x - (height // 2)), abs(y - (width // 2))
        # Calculate the blending factor
        blend_factor = min(1, distance / (max(height // 2, width // 2)))
        
        # Blend the colors
        blended_color = blend_colors(center_color, color, blend_factor)
        
        # Set the blended color in the output grid
        output_grid[x, y] = blended_color

    return output_grid

def blend_colors(color1, color2, factor):
    """
    Blend two colors based on the given factor.
    factor: 0.0 (fully color1), 1.0 (fully color2)
    """
    # Assuming colors are represented as strings
    return color1 if factor < 0.5 else color2