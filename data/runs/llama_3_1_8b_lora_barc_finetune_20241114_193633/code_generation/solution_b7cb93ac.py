from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, object extraction, color mapping

# description:
# In the input, you will see a grid containing multiple colored shapes, some of which are symmetric about their centers.
# To create the output, extract the largest symmetric shape and replace it with a new shape of a single color,
# while retaining the original colors of the other shapes.

def transform(input_grid):
    # Detect all the objects in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=False)
    largest_symmetric_object = None
    largest_area = 0

    # Check each object for symmetry and find the largest symmetric one
    for obj in objects:
        # Crop the object to isolate it
        cropped_obj = crop(obj, background=Color.BLACK)
        height, width = cropped_obj.shape
        
        # Check for horizontal symmetry
        is_symmetric = np.array_equal(cropped_obj, np.flipud(cropped_obj)) or np.array_equal(cropped_obj, np.fliplr(cropped_obj))
        
        # Calculate the area of the object
        area = np.sum(cropped_obj!= Color.BLACK)
        
        if is_symmetric and area > largest_area:
            largest_area = area
            largest_symmetric_object = cropped_obj
    
    # If we found a symmetric object, output its color
    if largest_symmetric_object is not None:
        # Get the color of the largest symmetric object
        output_color = largest_symmetric_object[0, 0]
        output_grid = np.full(input_grid.shape, Color.BLACK)
        # Find the position to place the new symmetric shape
        x_offset = (output_grid.shape[0] - largest_symmetric_object.shape[0]) // 2
        y_offset = (output_grid.shape[1] - largest_symmetric_object.shape[1]) // 2
        # Blit the largest symmetric object onto the output grid
        blit_sprite(output_grid, largest_symmetric_object, x=x_offset, y=y_offset, background=Color.BLACK)
        
        # Fill the rest of the grid with the output color
        output_grid[output_grid!= Color.BLACK] = output_color
    else:
        output_grid = np.full(input_grid.shape, Color.BLACK)  # If no symmetry, return a black grid

    return output_grid