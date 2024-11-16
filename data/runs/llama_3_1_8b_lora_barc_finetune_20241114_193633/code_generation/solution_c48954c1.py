from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern repetition, scaling, grid manipulation

# description:
# In the input, you will see a small pattern in the center of the grid. 
# To create the output, replicate the pattern in all four quadrants of the grid, 
# scaling it up by a factor of 2. The output grid should be large enough to contain 
# the scaled patterns in all quadrants.

def transform(input_grid):
    # Get the central pattern by finding the bounding box of the non-background pixels
    pattern = crop(input_grid)
    
    # Determine the size of the output grid
    scale_factor = 2
    output_height, output_width = pattern.shape[0] * scale_factor, pattern.shape[1] * scale_factor
    output_grid = np.full((output_height, output_width), Color.BLACK)

    # Calculate the center position for the output grid
    center_x = (output_width - pattern.shape[0] * scale_factor) // 2
    center_y = (output_height - pattern.shape[1] * scale_factor) // 2

    # Place the scaled pattern in all four quadrants
    blit_sprite(output_grid, pattern, x=center_x, y=center_y, background=Color.BLACK)  # Center
    blit_sprite(output_grid, pattern, x=center_x, y=center_y + pattern.shape[1] * scale_factor, background=Color.BLACK)  # Bottom
    blit_sprite(output_grid, pattern, x=center_x + pattern.shape[0] * scale_factor, y=center_y, background=Color.BLACK)  # Right
    blit_sprite(output_grid, pattern, x=center_x + pattern.shape[0] * scale_factor, y=center_y + pattern.shape[1] * scale_factor, background=Color.BLACK)  # Bottom right

    return output_grid