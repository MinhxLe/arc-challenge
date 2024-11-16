from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object detection, color transformation, grid splitting

# description:
# In the input, you will see a grid containing colored objects. Each object has a unique color, and the grid is divided into two halves vertically.
# To make the output grid, identify the color of the left half of the grid and replace all pixels in the right half with that color. 
# If the left half is completely black, leave the output unchanged.

def transform(input_grid):
    # Determine the dimensions of the grid
    height, width = input_grid.shape
    
    # Split the grid into left and right halves
    left_half = input_grid[:, :width // 2]
    right_half = input_grid[:, width // 2:]
    
    # Check if the left half is all black
    if np.all(left_half == Color.BLACK):
        return input_grid  # Return the original grid if left half is all black

    # Get the unique color of the left half
    unique_colors = np.unique(left_half)
    unique_colors = unique_colors[unique_colors!= Color.BLACK]  # Exclude black
    
    if len(unique_colors) == 0:
        raise ValueError("The left half contains no colors.")

    fill_color = unique_colors[0]  # Get the first unique color

    # Create the output grid by copying the left half and filling the right half
    output_grid = np.copy(input_grid)
    output_grid[:, width // 2:] = fill_color  # Fill the right half with the left half's color

    return output_grid