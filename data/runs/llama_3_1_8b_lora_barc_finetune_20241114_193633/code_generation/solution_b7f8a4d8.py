from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# rotation, color mapping, object detection

# description:
# In the input, you will see a grid with a central object of one color surrounded by a border of another color.
# To make the output, you should rotate the central object by 90 degrees clockwise and change its color to match the color of the border pixels.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # Find the central object and its color
    central_color = np.unique(input_grid[input_grid!= Color.BLACK])[0]  # Assuming the central object has a single color
    border_color = Color.BLACK  # Assuming the border is black by default

    # Find the connected components (the central object)
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True)
    central_object = objects[0]  # We assume there's only one central object

    # Get the bounding box of the central object
    x, y, width, height = bounding_box(central_object, background=Color.BLACK)

    # Extract the central object
    extracted_object = crop(central_object, background=Color.BLACK)

    # Rotate the extracted object 90 degrees clockwise
    rotated_object = np.rot90(extracted_object, -1)  # -1 for clockwise rotation

    # Get the position to place the rotated object
    output_x = x
    output_y = y

    # Change the color of the rotated object to match the border color
    rotated_object[rotated_object!= Color.BLACK] = border_color

    # Place the rotated object back into the output grid
    blit_object(output_grid, rotated_object, background=Color.BLACK)

    return output_grid