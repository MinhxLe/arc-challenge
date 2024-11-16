from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color extraction, pattern replication, grid transformation

# description:
# In the input, you will see a grid with a specific pattern made of colored pixels. 
# The pattern consists of a central shape surrounded by a border of pixels of a different color.
# To make the output, you should extract the central pattern, 
# and replicate it in a new grid, replacing the surrounding border pixels with a new color.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Crop the central pattern out from the input grid
    central_pattern = crop(input_grid, background=Color.BLACK)
    
    # Determine the size of the new output grid
    new_height, new_width = central_pattern.shape
    
    # Create an output grid filled with the new color (not black)
    output_grid = np.full((new_height, new_width), Color.BLACK)

    # Place the central pattern into the output grid
    blit_sprite(output_grid, central_pattern, x=0, y=0)

    return output_grid