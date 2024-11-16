from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# color extraction, pattern repetition, grid manipulation

# description:
# In the input, you will see a grid with a repeating pattern of colored pixels. 
# To create the output, extract the unique colors from the pattern and fill a new grid 
# with these colors in the same order they appear, repeating the pattern until the grid is filled.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True)
    
    # Get unique colors from the objects
    unique_colors = set()
    for obj in objects:
        unique_colors.update(np.unique(obj[obj!= Color.BLACK]))
    
    # Convert to a sorted list to maintain consistent order
    unique_colors = sorted(list(unique_colors))
    
    # Calculate the size of the output grid
    output_height = input_grid.shape[0]
    output_width = input_grid.shape[1]
    
    # Create the output grid
    output_grid = np.full((output_height, output_width), Color.BLACK)
    
    # Fill the output grid with the unique colors in a repeating pattern
    pattern_length = len(unique_colors)
    for i in range(output_height):
        for j in range(output_width):
            output_grid[i, j] = unique_colors[j % pattern_length]
    
    return output_grid