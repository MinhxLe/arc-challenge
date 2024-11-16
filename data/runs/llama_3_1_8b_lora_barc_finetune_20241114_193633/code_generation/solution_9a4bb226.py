from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object extraction, color transformation

# description:
# In the input, you will see a grid with several objects of different colors. 
# To create the output, extract the largest object and fill it with a single color that is the average color of all its pixels.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find all connected components in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)
    
    # Identify the largest object by the number of pixels
    largest_object = max(objects, key=lambda obj: np.sum(obj!= Color.BLACK))
    
    # Compute the average color of the largest object
    non_background_pixels = largest_object[largest_object!= Color.BLACK]
    average_color = np.mean(non_background_pixels).astype(int)

    # Create the output grid filled with the average color
    output_grid = np.full(largest_object.shape, average_color)

    # Blit the largest object onto the output grid
    blit_object(output_grid, largest_object, background=Color.BLACK)

    return output_grid