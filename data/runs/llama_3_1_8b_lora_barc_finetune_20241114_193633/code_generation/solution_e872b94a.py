from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color extraction, vertical stacking

# description:
# In the input, you will see a grid with a colored object surrounded by a background color.
# The task is to extract the colored object and create a new output grid that stacks the object vertically 
# for a specified number of times based on its height.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid to avoid modifying the original
    output_grid = np.copy(input_grid)

    # Detect the colored object in the input grid
    colored_objects = detect_objects(grid=output_grid, colors=Color.NOT_BLACK, monochromatic=False, connectivity=4)
    assert len(colored_objects) == 1, "There should be exactly one colored object."

    # Extract the object
    colored_object = colored_objects[0]

    # Crop the object to get the bounding box
    cropped_object = crop(colored_object, background=Color.BLACK)

    # Determine the height of the object
    object_height = np.sum(cropped_object!= Color.BLACK)

    # Create the output grid
    output_height = object_height * 3  # Stack the object 3 times vertically
    output_grid = np.full((output_height, cropped_object.shape[1]), Color.BLACK)

    # Stack the cropped object vertically in the output grid
    for i in range(3):
        blit_sprite(output_grid, cropped_object, x=0, y=i * object_height)

    return output_grid