from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry, reflection, color transformation

# description:
# In the input, you will see a grid filled with colored pixels. The colors can be any of the ten colors.
# To create the output grid, reflect the grid along its vertical axis and then replace all pixels that are 
# part of the original object with a new color (Color.RED) while keeping the background unchanged.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid for the output
    output_grid = np.copy(input_grid)

    # Reflect the grid along the vertical axis
    reflected_grid = np.fliplr(output_grid)

    # Define the color to replace the original object pixels
    new_color = Color.RED

    # Replace original object pixels with the new color
    output_grid = np.where(output_grid!= Color.BLACK, new_color, Color.BLACK)

    # Overlay the reflected grid onto the output grid
    output_grid = np.maximum(output_grid, reflected_grid)

    return output_grid