from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object movement, boundary checking, color transformation

# description:
# In the input, you will see a grid with a colored object that can move left, right, up, or down. 
# To make the output, the object will move to the nearest empty space while maintaining its original color.
# If the object touches the edge of the grid, it will bounce back to the last valid position.

def transform(input_grid):
    output_grid = np.copy(input_grid)
    background = Color.BLACK
    
    # Find the coordinates of the colored object
    object_coords = np.argwhere(input_grid!= Color.BLACK)

    if len(object_coords) == 0:
        return output_grid  # If no object found, return the input grid as output

    # Determine the current position of the object
    current_x, current_y = object_coords[0]

    # Get the color of the object
    object_color = input_grid[current_x, current_y]

    # Define possible movements (up, down, left, right)
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Check each possible movement
    for dx, dy in movements:
        new_x, new_y = current_x + dx, current_y + dy
        if 0 <= new_x < output_grid.shape[0] and 0 <= new_y < output_grid.shape[1]:
            if output_grid[new_x, new_y] == background:
                # If the new position is empty, move the object there
                output_grid[new_x, new_y] = object_color
                output_grid[current_x, current_y] = background
                return output_grid

    # If no valid space found, the object should bounce back to the last valid position
    output_grid[current_x, current_y] = object_color

    return output_grid