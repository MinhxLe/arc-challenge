from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color grouping, grid transformation, counting, symmetry

# description:
# In the input, you will see a grid filled with various colored pixels. The grid contains groups of connected pixels of the same color, separated by black pixels. 
# The goal is to transform the input grid by replacing each connected component of the same color with a single pixel of that color, 
# and the output grid will be the smallest bounding box that contains all these pixels. 
# If a color appears only once, it should not appear in the output.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find connected components in the input grid
    components = find_connected_components(input_grid, monochromatic=False, connectivity=4, background=Color.BLACK)
    
    # Dictionary to count occurrences of each color
    color_count = {}
    
    # Count occurrences of each color in the components
    for component in components:
        color = component[0, 0]  # All pixels in the component have the same color
        if color!= Color.BLACK:
            if color not in color_count:
                color_count[color] = 0
            color_count[color] += 1
            
    # Create the output grid
    output_grid = np.full((0, 0), Color.BLACK)
    
    # Create the bounding box for the output grid based on the colors counted
    for color, count in color_count.items():
        if count > 0:
            # Add the pixel to the output grid
            output_grid = np.concatenate((output_grid, np.full((1, 1), color)), axis=0)
            output_grid = np.concatenate((output_grid, np.full((1, 1), Color.BLACK)), axis=0)
    
    # Crop to the smallest bounding box
    output_grid = crop(output_grid, background=Color.BLACK)

    return output_grid