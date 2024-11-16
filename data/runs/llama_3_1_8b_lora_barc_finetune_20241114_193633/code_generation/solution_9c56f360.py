from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# symmetry, color matching, filling, mirroring

# description:
# In the input grid, you will see a colored object on a black background. The object has a distinct color pattern.
# To create the output grid, mirror the colored object across the vertical axis and fill in the mirrored part with the same color.

def transform(input_grid):
    # Plan:
    # 1. Extract the object from the input grid
    # 2. Mirror the object across the vertical axis
    # 3. Fill the mirrored part with the same color as the original object

    # Step 1: Find the connected component that represents the object
    objects = find_connected_components(input_grid, background=Color.BLACK)
    assert len(objects) == 1, "There should be exactly one object in the input grid"
    
    original_object = objects[0]

    # Step 2: Mirror the object across the vertical axis
    mirrored_object = original_object[:, ::-1]

    # Step 3: Create the output grid and fill the mirrored part
    output_grid = np.copy(input_grid)

    # Get the color of the original object
    object_color = np.unique(original_object[original_object!= Color.BLACK])[0]

    # Fill the mirrored part with the same color
    mirrored_x_start = output_grid.shape[1] // 2  # Center for mirroring
    output_grid[:, mirrored_x_start:] = np.where(mirrored_object!= Color.BLACK, mirrored_object, Color.BLACK)

    return output_grid