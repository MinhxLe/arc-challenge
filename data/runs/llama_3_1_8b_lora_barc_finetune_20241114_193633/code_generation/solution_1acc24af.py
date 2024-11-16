from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, overlapping shapes, transparency

# description:
# In the input, you will see two overlapping shapes of different colors. The first shape is a solid color,
# and the second shape is semi-transparent. To create the output, blend the colors of the overlapping region 
# where the two shapes meet, producing a new color that is a mix of the two colors.

def blend_colors(color1, color2):
    # Simple color blending function that averages the RGB values of the colors
    # Assuming colors are represented as integers (0-9)
    return (color1 + color2) // 2

def transform(input_grid):
    output_grid = np.copy(input_grid)
    
    # Get the colors of the shapes
    color1 = Color.BLUE  # Assume color1 is blue for the solid shape
    color2 = Color.GRAY  # Assume color2 is gray for the semi-transparent shape
    
    # Find the overlapping region
    overlap_mask = (input_grid == color1) & (input_grid == color2)
    
    # Blend colors where the two shapes overlap
    output_grid[overlap_mask] = Color.RED  # Let's assume blending results in red for simplicity

    return output_grid