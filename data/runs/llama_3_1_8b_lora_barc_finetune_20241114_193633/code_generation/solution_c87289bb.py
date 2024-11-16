from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color transformation, symmetry, mirroring

# description:
# In the input, you will see a grid with a colored shape and a mirror line of a different color. 
# To make the output, reflect the shape across the mirror line while ensuring that the resulting shape 
# maintains its original color. The output grid should be the same size as the input grid.

def transform(input_grid):
    # Create a copy of the input grid to work on
    output_grid = np.copy(input_grid)

    # Identify the color of the mirror line (the second color in the grid)
    unique_colors = np.unique(input_grid)
    mirror_color = unique_colors[1] if len(unique_colors) > 1 else Color.BLACK

    # Find the position of the mirror line
    mirror_line_positions = np.where(input_grid == mirror_color)
    
    # Get the coordinates of the mirror line
    for x in range(mirror_line_positions[0].shape[0]):
        for y in range(mirror_line_positions[1].shape[1]):
            if mirror_line_positions[0][x]!= -1 and mirror_line_positions[1][y]!= -1:
                mirror_x = mirror_line_positions[0][x]
                mirror_y = mirror_line_positions[1][y]

                # Reflect the shape across the mirror line
                for x_shape in range(input_grid.shape[0]):
                    for y_shape in range(input_grid.shape[1]):
                        if input_grid[x_shape, y_shape]!= Color.BLACK and (x_shape, y_shape)!= (mirror_x, mirror_y):
                            # Calculate reflected coordinates
                            reflected_x = mirror_x - (mirror_x - x_shape)
                            reflected_y = mirror_y - (y_shape - mirror_y)

                            # Ensure we stay within bounds
                            if 0 <= reflected_x < input_grid.shape[0] and 0 <= reflected_y < input_grid.shape[1]:
                                output_grid[reflected_x, reflected_y] = input_grid[x_shape, y_shape]

    return output_grid