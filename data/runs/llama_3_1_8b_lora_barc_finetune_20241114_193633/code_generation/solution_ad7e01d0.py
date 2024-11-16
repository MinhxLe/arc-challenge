from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# rotation, replication, symmetry

# description:
# In the input, you will see a grid with a single colored shape. To create the output, replicate this shape in all four quadrants of the grid, 
# rotating it 90 degrees clockwise for each quadrant. The output grid should be large enough to accommodate the shape in all four quadrants.

def transform(input_grid):
    # Find the shape in the input grid
    objects = find_connected_components(input_grid, connectivity=4, monochromatic=True)
    
    # Assuming there is only one shape
    shape = objects[0]

    # Get the bounding box of the shape
    x, y, w, h = bounding_box(shape)

    # Create an output grid that is large enough to hold the shape in all four quadrants
    output_size = max(w, h) * 2
    output_grid = np.full((output_size, output_size), Color.BLACK)

    # Define the rotations for the quadrants
    rotations = [shape, np.rot90(shape, k=1), np.rot90(shape, k=2), np.rot90(shape, k=3)]

    # Place each rotated shape in the corresponding quadrant
    for i, rot in enumerate(rotations):
        blit_sprite(output_grid, rot, x=(i % 2) * w, y=(i // 2) * h, background=Color.BLACK)

    return output_grid