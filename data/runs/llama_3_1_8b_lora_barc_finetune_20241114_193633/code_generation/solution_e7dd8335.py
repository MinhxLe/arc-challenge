from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# symmetry detection, color transformation

# description:
# In the input, you will see a grid with a pattern that exhibits rotational symmetry. 
# To make the output, identify the center of the pattern and replace the original colors with new colors 
# while preserving the symmetry around the center.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create an output grid initialized to the background color
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Detect the center of the pattern
    center_x, center_y = input_grid.shape[0] // 2, input_grid.shape[1] // 2

    # Identify the colors present in the input grid, excluding the background
    unique_colors = np.unique(input_grid[input_grid!= Color.BLACK])
    if len(unique_colors) == 0:
        return output_grid  # No colors to transform

    # Create a mapping from original colors to new colors
    color_map = {original_color: new_color for original_color in unique_colors for new_color in unique_colors if original_color!= new_color}

    # Apply the color transformation while preserving rotational symmetry
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            if input_grid[x, y] in color_map:
                output_grid[x, y] = color_map[input_grid[x, y]]
                # Calculate the symmetrical positions based on the center
                sym_x = center_x + (center_x - x)
                sym_y = center_y + (center_y - y)
                if 0 <= sym_x < output_grid.shape[0] and 0 <= sym_y < output_grid.shape[1]:
                    output_grid[sym_x, sym_y] = color_map[input_grid[x, y]]

    return output_grid