from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object detection, color mapping, size sorting

# description:
# In the input, you will see a grid filled with various colored objects on a black background. Each object has a unique color.
# To create the output, count the number of pixels in each object and sort the objects by their pixel count from smallest to largest.
# Then, create a new grid where each object is represented by its original color, but the colors are shifted in order of their size.

def transform(input_grid):
    # Find all connected components (objects) in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True)
    
    # Sort objects by the number of pixels (size)
    objects = sorted(objects, key=lambda obj: np.sum(obj!= Color.BLACK), reverse=True)

    # Create an output grid of the same shape as the input
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Draw each object in the output grid, with colors shifted according to their size
    for index, obj in enumerate(objects):
        # Get the color of the object (assuming monochromatic)
        color = obj[obj!= Color.BLACK][0]  # Get the color of the first non-black pixel
        # Get the bounding box of the object
        x, y, width, height = bounding_box(obj, background=Color.BLACK)
        
        # Calculate the position to blit the object in the output grid
        output_x = index % 10  # Shift by 1 for each object
        output_y = (index // 10) * 3  # Shift by 3 rows for each 10 objects

        # Place the object in the output grid
        blit_sprite(output_grid, obj, x=output_x, y=output_y, background=Color.BLACK)

    return output_grid