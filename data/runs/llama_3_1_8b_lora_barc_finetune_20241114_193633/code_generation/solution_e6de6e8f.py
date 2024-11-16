from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# counting, shape transformation, color mapping

# description:
# In the input, you will see a grid containing various colored shapes. 
# The output grid should represent the number of shapes of each color on a horizontal row:
# the first row should represent red shapes, the second for blue shapes, and so on, 
# with each shape represented by a filled square of that color in the output grid.
# Each shape is represented by a connected component in the input grid.

def transform(input_grid):
    # Detect all the connected components (shapes) in the input grid.
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True, connectivity=4)

    # Create a dictionary to count the number of shapes per color.
    color_count = {Color.RED: 0, Color.BLUE: 0, Color.GREEN: 0, Color.YELLOW: 0,
                   Color.GRAY: 0, Color.PINK: 0, Color.ORANGE: 0, Color.PURPLE: 0, 
                   Color.BROWN: 0, Color.BLACK: 0}

    # Count the shapes by color
    for obj in objects:
        color = obj[0, 0]  # Get the color of the shape
        color_count[color] += 1

    # Create output grid, initialized with black background
    output_grid = np.zeros((10, 10), dtype=int)  # 10x10 grid for output

    # Fill the output grid based on the counts
    current_row = 0
    for color, count in color_count.items():
        # Fill the output grid with squares of the specified color
        for i in range(count):
            if current_row + 1 >= output_grid.shape[0]:  # Check if we can fill any more rows
                break
            draw_line(output_grid, x=0, y=current_row, length=1, color=color, direction=(1, 0))  # Draw a vertical line
            draw_line(output_grid, x=0, y=current_row, length=1, color=color, direction=(0, 1))  # Draw a horizontal line
            current_row += 1

    return output_grid