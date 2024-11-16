from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# rotation, scaling

# description:
# In the input, you will see a square grid containing a pattern made up of colored pixels.
# To make the output, rotate the pattern by 90 degrees clockwise and scale it up by a factor of 2, 
# while maintaining the color arrangement.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Step 1: Rotate the input grid 90 degrees clockwise
    rotated_grid = np.rot90(input_grid, k=-1)

    # Step 2: Scale the rotated grid by a factor of 2
    output_grid = np.repeat(rotated_grid, 2, axis=0)
    output_grid = np.repeat(output_grid, 2, axis=1)

    return output_grid