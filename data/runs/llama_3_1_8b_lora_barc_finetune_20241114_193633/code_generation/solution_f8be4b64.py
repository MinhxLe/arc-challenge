from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color separation, object detection, surrounding color filling

# description:
# In the input, you will see a grid with multiple objects of different colors. 
# To create the output, you should identify the largest object by area, fill it with a new color, 
# and surround it with a contrasting color (the opposite color of the largest object).
# The output grid should contain the largest object filled with its own color and surrounded by the contrasting color.

def transform(input_grid):
    # Step 1: Detect all objects in the grid
    objects = detect_objects(grid=input_grid, monochromatic=False, connectivity=4, allowed_dimensions=[(1, 1), (2, 2), (3, 3), (4, 4)], colors=Color.NOT_BLACK)

    # Step 2: Find the largest object by area
    largest_object = max(objects, key=lambda obj: obj.size)

    # Step 3: Get the color of the largest object
    largest_color = largest_object[0, 0]  # Assuming monochromatic for the largest object

    # Step 4: Find the contrasting color (the opposite color)
    contrasting_color = Color.BLACK if largest_color!= Color.BLACK else Color.BLUE

    # Step 5: Create output grid and fill the largest object
    output_grid = np.full_like(input_grid, Color.BLACK)
    x, y, width, height = bounding_box(largest_object, background=Color.BLACK)
    output_grid[x:x + largest_object.shape[0], y:y + largest_object.shape[1]] = largest_object

    # Step 6: Surround the largest object with the contrasting color
    for i in range(output_grid.shape[0]):
        for j in range(output_grid.shape[1]):
            if output_grid[i, j] == largest_color:
                # Surround the object with contrasting color
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        if 0 <= i + dx < output_grid.shape[0] and 0 <= j + dy < output_grid.shape[1]:
                            output_grid[i + dx, j + dy] = contrasting_color

    return output_grid