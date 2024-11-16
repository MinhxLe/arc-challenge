from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# shape detection, boundary detection, cropping

# description:
# In the input, you will see a grid containing a single shape made of colored pixels. 
# The shape can be of various forms but should have a clear boundary defined by a different color (e.g., black).
# To create the output, crop the grid so that it only includes the area of the shape, removing all background pixels.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Detect the shape in the grid
    shape_objects = detect_objects(grid=input_grid, colors=Color.NOT_BLACK, monochromatic=False, connectivity=4, allowed_dimensions=None)
    
    # Find the largest connected component (the shape)
    shape = max(shape_objects, key=lambda obj: np.sum(obj!= Color.BLACK))

    # Crop the shape from the input grid
    output_grid = crop(shape, background=Color.BLACK)

    return output_grid