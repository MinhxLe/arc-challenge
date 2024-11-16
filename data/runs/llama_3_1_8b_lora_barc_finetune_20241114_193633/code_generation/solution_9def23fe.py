from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern extraction, boundary detection, color replacement

# description:
# In the input, you will see a grid with a complex pattern surrounded by a black border. 
# The pattern consists of colored pixels. To create the output, you should extract the pattern 
# and replace its color with a specified color (e.g., blue). The output grid should be the same size as the input grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid for output
    output_grid = np.copy(input_grid)

    # Find the connected components in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)

    # Extract the largest object, which is the pattern
    pattern = max(objects, key=lambda obj: obj.size)

    # Get the bounding box of the pattern
    x, y, width, height = bounding_box(pattern)

    # Crop the pattern from the grid
    cropped_pattern = crop(pattern, background=Color.BLACK)

    # Replace the color of the cropped pattern with blue
    cropped_pattern[cropped_pattern!= Color.BLACK] = Color.BLUE

    # Place the modified pattern back into the output grid
    blit_sprite(output_grid, cropped_pattern, x=x, y=y, background=Color.BLACK)

    return output_grid