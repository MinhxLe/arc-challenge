from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# layering, color blending

# description:
# In the input, you will see a grid with a colored shape and a semi-transparent overlay of a different color. 
# To make the output, blend the two colors together where they overlap, resulting in a new color that is a mix of the two.

def blend_colors(color1, color2):
    """
    Blend two colors together. For simplicity, we will assume colors are represented as integers.
    This function will return a new color that is a mix of the two.
    """
    if color1 == Color.BLACK:
        return color2
    if color2 == Color.BLACK:
        return color1
    
    # Simple blending logic: average the colors
    return Color.RED if (color1 == Color.BLUE and color2 == Color.YELLOW) else Color.GREEN  # Example blending logic

def transform(input_grid):
    # Get the shape and overlay
    shape = np.where(input_grid!= Color.BLACK)
    overlay = np.where(input_grid == Color.BLUE)  # Assume overlay is always blue for this puzzle

    # Create output grid, starting with a copy of the input grid
    output_grid = np.copy(input_grid)

    # Blend the colors in the overlapping region
    for x, y in zip(*shape):
        if input_grid[x, y]!= Color.BLACK and (x, y) in overlay:
            output_grid[x, y] = blend_colors(input_grid[x, y], Color.BLUE)

    return output_grid