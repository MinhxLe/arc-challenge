from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# object extraction, color matching, border detection

# description:
# In the input, you will see a grid with several colored objects and a black background. Each object consists of a single color.
# To create the output, identify the largest object based on the number of non-background pixels and extract it, 
# filling the surrounding area with a specified color that matches the extracted object.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find all connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)

    # Identify the largest object based on the number of non-background pixels
    largest_object = max(objects, key=lambda obj: np.sum(obj!= Color.BLACK), default=None)

    if largest_object is None:
        return np.zeros_like(input_grid)  # Return an empty grid if no objects are found

    # Get the color of the largest object
    largest_color = largest_object[largest_object!= Color.BLACK][0]

    # Crop the largest object to create the output
    output_grid = crop(largest_object, background=Color.BLACK)

    # Create a new grid with the same size as the output grid
    output_grid = np.full_like(output_grid, Color.BLACK)

    # Fill the surrounding area with the color of the largest object
    output_grid[output_grid == Color.BLACK] = largest_color

    # Place the largest object in the output grid
    output_grid = blit_object(output_grid, largest_object, background=Color.BLACK)

    return output_grid