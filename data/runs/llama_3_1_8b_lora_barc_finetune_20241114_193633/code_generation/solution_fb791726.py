from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# pattern replication, symmetry

# description:
# In the input, you will see a grid with a central pattern of colored pixels. 
# To create the output, replicate the central pattern in all four quadrants of the grid, 
# maintaining the symmetry of the original pattern.

def transform(input_grid):
    # Crop the central pattern from the input grid
    central_pattern = crop(input_grid, background=Color.BLACK)
    
    # Get the dimensions of the central pattern
    pattern_height, pattern_width = central_pattern.shape
    
    # Create an output grid that is large enough to hold the pattern in all quadrants
    output_height = pattern_height * 2
    output_width = pattern_width * 2
    output_grid = np.full((output_height, output_width), Color.BLACK)

    # Place the central pattern in all four quadrants
    blit_sprite(output_grid, central_pattern, x=0, y=0, background=Color.BLACK)  # Top-left
    blit_sprite(output_grid, central_pattern, x=0, y=pattern_height, background=Color.BLACK)  # Bottom-left
    blit_sprite(output_grid, central_pattern, x=pattern_width, y=0, background=Color.BLACK)  # Top-right
    blit_sprite(output_grid, central_pattern, x=pattern_width, y=pattern_height, background=Color.BLACK)  # Bottom-right

    return output_grid