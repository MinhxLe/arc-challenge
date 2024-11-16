from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color transformation, mirroring, symmetry

# description:
# In the input, you will see a grid containing several colored shapes. 
# To create the output, mirror the grid vertically and change the color of each shape to the color of the pixel in the original grid that is opposite to it on the vertical axis.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create an output grid with the same shape as the input
    output_grid = np.copy(input_grid)
    
    # Get the number of rows in the grid
    num_rows = input_grid.shape[0]
    
    # Iterate through each row to mirror and change colors
    for i in range(num_rows):
        # Get the color of the current row
        current_row = input_grid[i]
        
        # Mirror the row
        mirrored_row = current_row[::-1]

        # Replace the original row with the mirrored row
        output_grid[i] = mirrored_row

        # Change the colors to the corresponding colors from the opposite row
        for j in range(len(mirrored_row)):
            output_grid[i][j] = input_grid[num_rows - 1 - i][j]

    return output_grid