from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, symmetry detection

# description:
# In the input, you will see a grid with colored pixels arranged in a symmetric pattern around a center point. 
# The center point is defined as the center of the grid. 
# To make the output, you should blend the colors of the pixels in the four quadrants of the grid, 
# replacing the colors in the output grid with the average color of the corresponding quadrants.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)
    height, width = input_grid.shape
    mid_x, mid_y = height // 2, width // 2
    
    # Calculate the center point
    center_x, center_y = mid_x, mid_y

    # Define the quadrants
    quadrants = {
        'top_left': input_grid[:mid_x, :mid_y],
        'top_right': input_grid[:mid_x, mid_y:],
        'bottom_left': input_grid[mid_x:, :mid_y],
        'bottom_right': input_grid[mid_x:, mid_y:]
    }

    # Create an output grid initialized to the background color
    output_grid = np.full((height, width), Color.BLACK)

    # Blend the colors in each quadrant
    for quadrant_name, quadrant in quadrants.items():
        if np.any(quadrant!= Color.BLACK):
            # Calculate the average color of the quadrant
            unique_colors, counts = np.unique(quadrant[quadrant!= Color.BLACK], return_counts=True)
            average_color = unique_colors[np.argmax(counts)]  # Most frequent color in the quadrant
            # Set the output grid with the average color in the corresponding quadrant
            if quadrant_name == 'top_left':
                output_grid[:mid_x, :mid_y] = average_color
            elif quadrant_name == 'top_right':
                output_grid[:mid_x, mid_y:] = average_color
            elif quadrant_name == 'bottom_left':
                output_grid[mid_x:, :mid_y] = average_color
            elif quadrant_name == 'bottom_right':
                output_grid[mid_x:, mid_y:] = average_color

    return output_grid