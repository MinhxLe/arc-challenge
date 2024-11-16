from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel transformation, rotation, color transformation

# description:
# In the input, you will see a grid with a colored object and a set of colored pixels surrounding it. 
# To create the output, rotate the object 90 degrees clockwise and change its color to the color of the nearest surrounding pixel.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    # Find all connected components in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)

    for obj in objects:
        # Get the bounding box of the object
        x, y, width, height = bounding_box(obj)

        # Crop the object from the grid
        object_sprite = crop(obj, background=Color.BLACK)

        # Rotate the object 90 degrees clockwise
        rotated_object = np.rot90(object_sprite, k=-1)

        # Get the color of the nearest surrounding pixel
        surrounding_colors = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                for x_check, y_check in np.argwhere(input_grid!= Color.BLACK):
                    if (x_check - x < width and y_check - y < height) and \
                       (x_check + dx >= 0 and x_check < input_grid.shape[0] and
                        y_check + dy >= 0 and y_check < input_grid.shape[1]):
                        surrounding_colors.append(input_grid[x_check, y_check])
        
        # Find the nearest color from surrounding colors
        if surrounding_colors:
            nearest_color = max(set(surrounding_colors), key=lambda color: np.count_nonzero(input_grid == color))
        else:
            nearest_color = Color.BLACK  # Default to black if no surrounding colors

        # Blit the rotated object into the output grid
        blit_sprite(output_grid, rotated_object, x=x, y=y, background=Color.BLACK)

        # Change the color of the rotated object to the nearest surrounding color
        for x_check, y_check in np.argwhere(rotated_object!= Color.BLACK):
            output_grid[y + x_check, x + y_check] = nearest_color

    return output_grid