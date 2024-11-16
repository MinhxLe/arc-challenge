from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel manipulation, boundary detection, color transformation

# description:
# In the input, you will see a grid filled with colored pixels, with a single black boundary pixel surrounding the colored area. 
# To make the output, transform the colored area by changing its color based on its distance from the boundary. 
# Pixels closer to the boundary should become darker shades of their original color, while pixels further away remain the same color.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)
    boundary_mask = (input_grid == Color.BLACK)

    # Find the coordinates of the boundary pixels
    boundary_coords = np.argwhere(boundary_mask)

    # Get the unique colors of the colored area (excluding the background)
    unique_colors = np.unique(input_grid[input_grid!= Color.BLACK])
    
    # Transform colors based on distance from the boundary
    for x, y in np.argwhere(input_grid!= Color.BLACK):
        distance = 0
        # Calculate the Manhattan distance from the nearest boundary pixel
        for bx, by in boundary_coords:
            distance += abs(x - bx) + abs(y - by)
        
        # Normalize distance to a range of 0 to 1 (closer to the boundary gets closer to black)
        distance = max(0, min(1, distance / (x + y + 1)))  # Normalize using height + width + 1 to avoid division by zero
        color_index = int(distance * len(unique_colors))
        new_color = unique_colors[color_index]
        output_grid[x, y] = new_color

    return output_grid