from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# pattern extraction, color propagation

# description:
# In the input, you will see a grid with a central pattern made up of colored pixels and a border of a single color (e.g., gray). 
# To create the output, you should identify the central pattern and replicate it to fill the entire grid while ensuring that the original border color is preserved in the corners.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Detect the bounding box of the non-background pixels (the pattern)
    pattern_objects = detect_objects(grid=input_grid, background=Color.GRAY, monochromatic=True, connectivity=4, allowed_dimensions=None)
    assert len(pattern_objects) == 1  # We expect exactly one pattern

    # Get the pattern and its bounding box
    pattern = crop(pattern_objects[0], background=Color.GRAY)
    x, y, w, h = bounding_box(pattern)

    # Create the output grid filled with the original border color
    output_grid = np.full(input_grid.shape, Color.GRAY)

    # Fill the output grid with the pattern
    for i in range(0, output_grid.shape[0], h):
        for j in range(0, output_grid.shape[1], w):
            blit_sprite(output_grid, pattern, x=j, y=i, background=Color.GRAY)

    return output_grid