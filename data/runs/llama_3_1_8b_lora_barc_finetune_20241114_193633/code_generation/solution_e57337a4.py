from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, boundary detection, object transformation

# description:
# In the input, you will see a grid filled with various colors, and some pixels are black (background).
# To make the output, you should identify the largest connected component (object) that has a non-black color.
# The output should be a grid that contains only this object, with all other pixels set to black.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find the largest connected component (object) in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)
    largest_object = max(objects, key=lambda obj: np.sum(obj!= Color.BLACK))

    # Crop the largest object to remove excess black pixels
    output_grid = crop(largest_object, background=Color.BLACK)

    return output_grid