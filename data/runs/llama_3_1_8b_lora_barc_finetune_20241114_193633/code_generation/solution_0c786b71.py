from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# mirroring, pattern replication

# description:
# In the input, you will see a pattern of colored pixels on a black background. 
# To make the output, replicate the pattern on the opposite side of the grid by mirroring it across the center line, 
# ensuring that the mirrored pattern aligns perfectly with the original.

def transform(input_grid):
    # Crop the input grid to find the bounding box of the pattern
    cropped_pattern = crop(input_grid, background=Color.BLACK)

    # Get the dimensions of the cropped pattern
    pattern_height, pattern_width = cropped_pattern.shape

    # Create the output grid, which will be twice the width of the pattern
    output_grid = np.full((pattern_height, pattern_width * 2), Color.BLACK)

    # Place the original pattern on the left side of the output grid
    blit_sprite(output_grid, cropped_pattern, x=0, y=0, background=Color.BLACK)

    # Create a mirrored version of the cropped pattern
    mirrored_pattern = cropped_pattern[:, ::-1]

    # Place the mirrored pattern on the right side of the output grid
    blit_sprite(output_grid, mirrored_pattern, x=pattern_width, y=0, background=Color.BLACK)

    return output_grid