from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern detection, color mapping, symmetry

# description:
# In the input, you will see a grid with a pattern of colored pixels. The pattern consists of a central shape surrounded by a ring of differently colored pixels.
# To make the output, you should identify the central shape and replace it with a new shape of the same color, while keeping the surrounding ring of pixels intact.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid for output
    output_grid = np.copy(input_grid)

    # Find connected components in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)

    # Identify the central shape and surrounding ring
    for obj in objects:
        # Crop the object to get the bounding box
        x, y, w, h = bounding_box(obj, background=Color.BLACK)
        cropped_shape = crop(obj, background=Color.BLACK)

        # Get the color of the central shape
        shape_color = cropped_shape[0, 0]

        # Replace the central shape with the same color in the output grid
        output_grid[x:x + w, y:y + h] = shape_color

        # Define the ring of pixels around the shape
        ring_positions = [
            (x - 1, y), (x + w + 1, y), (x, y - 1), (x, y + h + 1),
            (x - 1, y - 1), (x + w + 1, y - 1), (x - 1, y + h + 1), (x + w + 1, y + h + 1)
        ]
        # Replace the ring with the surrounding colors
        for pos in ring_positions:
            if 0 <= pos[0] < output_grid.shape[0] and 0 <= pos[1] < output_grid.shape[1]:
                output_grid[pos] = Color.ORANGE  # Replace with a specific color for the ring

    return output_grid