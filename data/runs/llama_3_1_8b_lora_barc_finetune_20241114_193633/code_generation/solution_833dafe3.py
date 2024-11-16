from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern detection, mirroring, color transformation

# description:
# In the input, you will see a grid with a distinct pattern of colored pixels. 
# To create the output, mirror the pattern across both the vertical and horizontal axes, 
# creating a symmetrical design that reflects the original pattern in all quadrants.

def transform(input_grid):
    # Detect the pattern in the input grid
    pattern = crop(input_grid, background=Color.BLACK)
    
    # Get the dimensions of the pattern
    h, w = pattern.shape
    
    # Create an output grid that is twice the size of the pattern
    output_grid = np.full((h * 2, w * 2), Color.BLACK, dtype=int)

    # Place the original pattern in the top-left corner
    blit_sprite(output_grid, pattern, x=0, y=0, background=Color.BLACK)

    # Mirror the pattern to the top-right corner
    blit_sprite(output_grid, pattern, x=w, y=0, background=Color.BLACK)

    # Mirror the pattern to the bottom-left corner
    blit_sprite(output_grid, pattern, x=0, y=h, background=Color.BLACK)

    # Mirror the pattern to the bottom-right corner
    blit_sprite(output_grid, pattern, x=w, y=h, background=Color.BLACK)

    return output_grid