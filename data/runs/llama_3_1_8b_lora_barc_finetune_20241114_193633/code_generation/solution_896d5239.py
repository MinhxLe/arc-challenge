from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color replacement, pixel manipulation

# description:
# In the input, you will see a grid filled with a specific color pattern. 
# The grid contains a specific shape made up of two colors: one color forms the outline of the shape, 
# while another color fills the interior of the shape. 
# To create the output, replace the filling color with a new color, while keeping the outline color intact.

def transform(input_grid):
    # Identify the two colors used in the grid
    unique_colors = np.unique(input_grid)
    outline_color, fill_color = Color.BLACK, Color.BLUE  # Assuming the first two colors are the outline and fill colors
    
    # Create a copy of the input grid to modify
    output_grid = np.copy(input_grid)

    # Replace the fill color with a new color while keeping the outline color intact
    new_fill_color = Color.GREEN  # You can choose any color you want to replace the fill color with
    output_grid[output_grid == fill_color] = new_fill_color

    return output_grid