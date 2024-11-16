from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object extraction, background detection, color replacement

# description:
# In the input, you will see a grid with various colored shapes and a distinct background color.
# To create the output grid, extract all shapes that are not the background color and place them on a new grid
# while changing their colors to match the background color.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Identify the background color
    background_color = Color.BLACK

    # Find all connected components that are not the background color
    objects = find_connected_components(input_grid, background=background_color, monochromatic=False, connectivity=8)

    # Create an output grid of the same size as the input grid, filled with the background color
    output_grid = np.full_like(input_grid, background_color)

    # Iterate through each connected component and change their color to match the background
    for obj in objects:
        # Get the bounding box of the object
        x, y, width, height = bounding_box(obj, background=background_color)

        # Crop the object to extract it
        cropped_object = crop(obj, background=background_color)

        # Change the color of the cropped object to the background color
        cropped_object[cropped_object!= Color.BLACK] = background_color

        # Blit the cropped object to the output grid at the same position
        output_grid = blit_sprite(output_grid, cropped_object, x=x, y=y, background=background_color)

    return output_grid