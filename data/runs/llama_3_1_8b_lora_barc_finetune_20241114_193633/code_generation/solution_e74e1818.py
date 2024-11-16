from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, symmetry detection, transformation

# description:
# In the input, you will see a grid with several colored objects. Each object consists of one color, and the background is black.
# To create the output, identify the largest object, and replace its color with the average color of all other objects in the grid.
# The output grid should reflect this new color for the largest object.

def transform(input_grid):
    # Detect all objects in the input grid
    objects = detect_objects(grid=input_grid, background=Color.BLACK, monochromatic=True, connectivity=4, allowed_dimensions=None)
    
    # Find the largest object
    largest_object = max(objects, key=lambda obj: obj.size)
    
    # Get the color of the largest object
    largest_color = largest_object[0, 0]  # Assuming monochromatic objects
    
    # Calculate the average color of all other objects
    other_colors = [obj[0, 0] for obj in objects if obj!= largest_object]
    if other_colors:
        average_color = np.mean(other_colors)
    else:
        average_color = largest_color  # If there are no other colors, keep the largest color

    # Create the output grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Replace the color of the largest object with the average color
    output_grid[largest_object!= Color.BLACK] = average_color
    
    return output_grid