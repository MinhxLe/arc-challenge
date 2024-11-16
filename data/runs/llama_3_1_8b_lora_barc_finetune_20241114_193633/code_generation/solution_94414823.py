from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, color transformation

# description:
# In the input, you will see a grid with a pattern that has rotational symmetry, 
# but one of the colors is missing. To make the output, fill in the missing color 
# in the positions where the pattern is symmetrically mirrored.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid to manipulate
    output_grid = np.copy(input_grid)
    
    # Find the center of the grid
    center_x, center_y = input_grid.shape[0] // 2, input_grid.shape[1] // 2
    
    # Get all unique colors in the input grid excluding the background
    unique_colors = np.unique(input_grid)
    unique_colors = unique_colors[unique_colors!= Color.BLACK]
    
    # Check for each pixel if it has a color and its symmetric counterpart is missing
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            if input_grid[x, y]!= Color.BLACK:
                # Calculate the symmetric positions
                sym_positions = [
                    (x, y), 
                    (2 * center_x - x, y), 
                    (x, 2 * center_y - y), 
                    (2 * center_x - x, 2 * center_y - y),
                ]
                
                # Fill in the missing color in symmetric positions
                for sx, sy in sym_positions:
                    if 0 <= sx < input_grid.shape[0] and 0 <= sy < input_grid.shape[1]:
                        if output_grid[sx, sy] == Color.BLACK:
                            output_grid[sx, sy] = input_grid[x, y]

    return output_grid