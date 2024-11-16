from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# radial symmetry, color propagation

# description:
# In the input, you will see a colored pixel in the center of a black grid. 
# To make the output, you should draw a pattern radiating outwards from the central pixel in all directions, 
# using a different color for each quadrant. The colors should be drawn in the following order: 
# top, right, bottom, left, and repeat in the order of the quadrant quadrants.

def transform(input_grid):
    # Determine the dimensions of the grid
    height, width = input_grid.shape
    output_grid = np.full_like(input_grid, Color.BLACK)

    # Get the center of the grid
    center_x, center_y = height // 2, width // 2

    # Get the color of the central pixel
    central_color = input_grid[center_x, center_y]

    # Define the colors for each quadrant
    quadrant_colors = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW]

    # Draw the pattern in each quadrant
    for i in range(4):
        color = quadrant_colors[i]
        # Calculate the coordinates for the current quadrant
        if i == 0:  # Top quadrant
            draw_line(output_grid, center_x, center_y - 1, length=height - 1, direction=(0, 1), color=color)
            draw_line(output_grid, center_x + 1, center_y, length=width - 1, direction=(1, 0), color=color)
        elif i == 1:  # Right quadrant
            draw_line(output_grid, center_x, center_y + 1, length=height - 1, direction=(1, 0), color=color)
            draw_line(output_grid, center_x, center_y, length=width - 1, direction=(0, 1), color=color)
        elif i == 2:  # Bottom quadrant
            draw_line(output_grid, center_x + 1, center_y, length=height - 1, direction=(-1, 0), color=color)
            draw_line(output_grid, center_x, center_y - 1, length=width - 1, direction=(0, -1), color=color)
        else:  # Left quadrant
            draw_line(output_grid, center_x, center_y - 1, length=height - 1, direction=(-1, 1), color=color)
            draw_line(output_grid, center_x, center_y, length=width - 1, direction=(1, -1), color=color)

    # Fill in the central pixel
    output_grid[center_x, center_y] = central_color

    return output_grid