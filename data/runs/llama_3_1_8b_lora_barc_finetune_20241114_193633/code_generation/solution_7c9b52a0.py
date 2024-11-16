from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# merging, color blending, shape extraction

# description:
# In the input, you will see two overlapping colored shapes on a black background. 
# To create the output, merge the two shapes by averaging the colors of the overlapping pixels, 
# while leaving the non-overlapping parts intact.

def transform(input_grid):
    # Create a copy of the input grid to hold the output
    output_grid = np.copy(input_grid)

    # Find the connected components of the grid
    components = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)

    # Assume there are only two shapes
    assert len(components) == 2

    # Extract the two shapes
    shape1 = components[0]
    shape2 = components[1]

    # Get the bounding boxes of both shapes
    x1, y1, w1, h1 = bounding_box(shape1)
    x2, y2, w2, h2 = bounding_box(shape2)

    # Calculate the overlap region
    overlap_x = max(x1, x2)
    overlap_y = max(y1, y2)
    overlap_w = min(x1 + w1, x2 + w2) - overlap_x
    overlap_h = min(y1 + h1, y2 + h2) - overlap_y

    # Create a mask for the overlap area
    overlap_mask = np.zeros((overlap_h, overlap_w), dtype=int)

    # Fill the overlap_mask with the average color of the overlapping pixels
    for x in range(overlap_h):
        for y in range(overlap_w):
            if shape1[x + overlap_y - y1, y + overlap_x - x1]!= Color.BLACK and shape2[x + overlap_y - y1, y + overlap_x - x1]!= Color.BLACK:
                # Average the colors
                color1 = shape1[x + overlap_y - y1, y + overlap_x - x1]
                color2 = shape2[x + overlap_y - y1, y + overlap_x - x1]
                average_color = (color1 + color2) // 2  # Simple average (assuming colors are integers)
                overlap_mask[x, y] = average_color
            else:
                overlap_mask[x, y] = Color.BLACK

    # Place the overlap region in the output grid
    output_grid[overlap_y:overlap_y + overlap_h, overlap_x:overlap_x + overlap_w] = overlap_mask

    # Remove the shapes from the output grid
    output_grid[output_grid == Color.BLACK] = Color.BLACK

    return output_grid