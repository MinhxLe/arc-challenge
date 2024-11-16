from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, color filling, pattern completion

# description:
# In the input, you will see a grid with a pattern that is missing some sections, represented by black pixels.
# To create the output, fill in the missing sections of the pattern using the color of the surrounding pixels,
# ensuring that the filled sections maintain the symmetry of the original pattern.

def transform(input_grid):
    # Create a copy of the input grid to fill in the missing sections
    output_grid = np.copy(input_grid)

    # Find connected components in the grid
    objects = find_connected_components(output_grid, background=Color.BLACK, connectivity=8, monochromatic=False)

    # Iterate through each connected component
    for obj in objects:
        # Get the bounding box of the current object
        x, y, width, height = bounding_box(obj)

        # Determine the color of the surrounding pixels
        surrounding_colors = set()
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if (dx, dy)!= (0, 0):
                    neighbor_x = x + dx
                    neighbor_y = y + dy
                    if 0 <= neighbor_x < output_grid.shape[0] and 0 <= neighbor_y < output_grid.shape[1]:
                        surrounding_colors.add(output_grid[neighbor_x, neighbor_y])

        # Determine the color to fill in the missing sections
        fill_color = None
        for color in surrounding_colors:
            if color!= Color.BLACK:
                fill_color = color
                break

        # Fill in the black pixels within the bounding box of the object with the determined fill color
        for i in range(height):
            for j in range(width):
                if obj[i, j] == Color.BLACK:
                    output_grid[x + i, y + j] = fill_color

    return output_grid