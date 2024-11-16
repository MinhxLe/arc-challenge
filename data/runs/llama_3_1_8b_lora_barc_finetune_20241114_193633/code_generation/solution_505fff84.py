from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# object detection, rotation, color extraction

# description:
# In the input, you will see a grid containing various colored objects.
# To make the output, extract each object, rotate it 90 degrees clockwise, and arrange the rotated objects in a new grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Detect all connected components (objects) in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True)

    # Prepare the output grid based on the number of detected objects
    output_size = len(objects)
    output_grid = np.zeros((output_size, output_size), dtype=int)

    for index, obj in enumerate(objects):
        # Rotate the object 90 degrees clockwise
        rotated_obj = np.rot90(obj, k=-1)  # Rotate clockwise
        # Crop to remove any black borders
        cropped_obj = crop(rotated_obj, background=Color.BLACK)

        # Place the rotated object in the output grid
        blit_sprite(output_grid, cropped_obj, x=index, y=0, background=Color.BLACK)

    return output_grid