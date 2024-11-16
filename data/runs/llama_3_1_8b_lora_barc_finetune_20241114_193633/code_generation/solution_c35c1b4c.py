from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, color filling, color propagation

# description:
# In the input, you will see a grid with a shape that has rotational symmetry but has some pixels missing (black).
# To create the output, fill in the missing pixels to complete the shape while maintaining its rotational symmetry.

def transform(input_grid):
    # Step 1: Detect rotational symmetry
    sym = detect_rotational_symmetry(input_grid, ignore_colors=[Color.BLACK])
    
    # Step 2: Find all colored pixels in the input grid
    colored_pixels = np.argwhere(input_grid!= Color.BLACK)
    
    # Step 3: Fill in the missing pixels based on the rotational symmetry
    output_grid = np.copy(input_grid)
    for x, y in colored_pixels:
        # Get the color of the current pixel
        color = input_grid[x, y]
        
        # Compute the orbit of the current pixel under the detected symmetry
        orbit_pixels = orbit(input_grid, x, y, sym)
        
        # Fill in the missing pixels in the output grid
        for ox, oy in orbit_pixels:
            if output_grid[ox, oy] == Color.BLACK:  # Only fill if the pixel is black
                output_grid[ox, oy] = color
    
    return output_grid