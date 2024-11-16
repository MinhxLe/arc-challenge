from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# radial symmetry, color filling, pattern completion

# description:
# In the input, you will see a grid with a radial pattern that is partially filled with colored pixels.
# To make the output, fill in the missing sections of the pattern using the colors of the existing pixels in a radial symmetry.

def transform(input_grid):
    # Get the size of the input grid
    rows, cols = input_grid.shape
    
    # Create an output grid initialized to the background color
    output_grid = np.full((rows, cols), Color.BLACK)

    # Find the center of the grid
    center_x, center_y = rows // 2, cols // 2

    # Iterate over the grid to fill in the output grid
    for x in range(rows):
        for y in range(cols):
            if input_grid[x, y]!= Color.BLACK:
                # Calculate the radial symmetry points
                dx = x - center_x
                dy = y - center_y
                radius = int(np.sqrt(dx**2 + dy**2))
                
                # Fill in the radial symmetry points
                for angle in range(8):  # 8 directions (0, 1, 2,..., 7)
                    rad_angle = np.radians(angle)
                    new_x = int(center_x + radius * np.cos(rad_angle) + dx)
                    new_y = int(center_y + radius * np.sin(rad_angle) + dy)
                    if 0 <= new_x < rows and 0 <= new_y < cols:
                        output_grid[new_x, new_y] = input_grid[x, y]

    return output_grid