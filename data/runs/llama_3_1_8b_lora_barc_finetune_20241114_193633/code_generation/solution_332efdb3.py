from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, color filling

# description:
# In the input, you will see a grid with a pattern that has rotational symmetry. 
# To create the output, fill in the missing sections of the pattern to complete the symmetry, 
# using a contrasting color to highlight the filled sections.

def transform(input_grid):
    # Find the center of the grid
    center_x, center_y = input_grid.shape[0] // 2, input_grid.shape[1] // 2

    # Create an output grid initialized to the input grid
    output_grid = np.copy(input_grid)

    # Detect rotational symmetry
    sym = detect_rotational_symmetry(input_grid, ignore_colors=[Color.BLACK])
    
    # Iterate through all pixels in the grid
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            # If the pixel is not the background, we want to check its symmetric counterparts
            if input_grid[x, y]!= Color.BLACK:
                # Get the color of the current pixel
                current_color = input_grid[x, y]
                
                # Find the symmetries (rotational transformations)
                for i in range(1, 4):  # 1, 2, and 3 rotations
                    rotated_x, rotated_y = sym.apply(x, y, iters=i)
                    # Fill the missing pixels in the output grid
                    if output_grid[rotated_x, rotated_y] == Color.BLACK:
                        output_grid[rotated_x, rotated_y] = current_color

    # Change the color of the filled sections to a contrasting color (e.g., Color.BLUE)
    contrasting_color = Color.BLUE
    output_grid[output_grid == Color.BLACK] = contrasting_color

    return output_grid