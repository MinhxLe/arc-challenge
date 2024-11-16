from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# rotational symmetry, pattern replication

# description:
# In the input, you will see a pattern of colored pixels arranged in a circular layout. 
# To create the output, replicate this pattern in all four quadrants of the grid, 
# maintaining the same orientation and distance from the center, creating a complete circular pattern.

def transform(input_grid):
    # Find the connected components (the circular pattern)
    objects = find_connected_components(input_grid, monochromatic=True, connectivity=8)

    # Assuming there's only one circular pattern in the input
    assert len(objects) == 1
    circle_pattern = objects[0]

    # Get the bounding box of the circular pattern
    x, y, width, height = bounding_box(circle_pattern, background=Color.BLACK)

    # Create an output grid that is large enough to fit the pattern in all quadrants
    output_size = max(width, height) * 2
    output_grid = np.full((output_size, output_size), Color.BLACK)

    # Calculate the center position for placing the pattern
    center_x, center_y = output_size // 2, output_size // 2

    # Place the pattern in all four quadrants
    for dx in [0, width]:
        for dy in [0, height]:
            blit_sprite(output_grid, crop(circle_pattern), center_x + dx - x, center_y + dy - y, background=Color.BLACK)

    return output_grid