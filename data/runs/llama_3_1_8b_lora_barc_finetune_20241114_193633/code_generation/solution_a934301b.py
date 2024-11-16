from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# symmetry detection, object extraction, color replacement

# description:
# In the input, you will see a grid with various colored objects on a black background.
# The task is to identify any objects that exhibit rotational symmetry and extract them,
# placing them into a new grid with a transparent background.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Detect all objects in the input grid
    objects = detect_objects(grid=input_grid, background=Color.BLACK, connectivity=8, monochromatic=False, allowed_dimensions=None)

    # Create an output grid initialized to the background color (black)
    output_grid = np.zeros_like(input_grid)

    # Check each object for rotational symmetry
    for obj in objects:
        # Crop the object to remove excess black pixels
        cropped_obj = crop(obj, background=Color.BLACK)

        # Check if the object has rotational symmetry
        sym = detect_rotational_symmetry(cropped_obj, ignore_colors=[Color.BLACK])
        if sym:
            # If it does, blit the object onto the output grid at the position it was found
            x, y = object_position(obj, background=Color.BLACK)
            blit_sprite(output_grid, cropped_obj, x=x, y=y, background=Color.BLACK)

    return output_grid