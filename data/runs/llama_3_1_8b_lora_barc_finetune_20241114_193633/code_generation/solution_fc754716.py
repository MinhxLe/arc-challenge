from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern extraction, color transformation, grid manipulation

# description:
# In the input, you will see a grid with a specific color pattern surrounded by a background color. 
# The goal is to extract this pattern and transform it into a new grid where each pixel of the pattern is replaced 
# with the color of the pixel in the corresponding position in a specified reference grid. 
# The output grid will be the same size as the input grid.

def transform(input_grid: np.ndarray, reference_grid: np.ndarray) -> np.ndarray:
    # Find the connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=8, monochromatic=True)
    
    # Assume the first detected object is the one we want to transform
    pattern = objects[0]
    
    # Create an output grid of the same size as the input grid, initialized to the background color
    output_grid = np.full(input_grid.shape, Color.BLACK)
    
    # For each pixel in the pattern, find its corresponding position in the reference grid
    for x in range(pattern.shape[0]):
        for y in range(pattern.shape[1]):
            if pattern[x, y]!= Color.BLACK:  # Only process non-background pixels
                # Find the color in the reference grid at the same position
                reference_color = reference_grid[x, y]
                # Assign the reference color to the corresponding position in the output grid
                output_grid[x, y] = reference_color
                
    return output_grid