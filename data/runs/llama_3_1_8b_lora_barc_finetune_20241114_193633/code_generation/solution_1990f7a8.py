from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel aggregation, color transformation, grid manipulation

# description:
# In the input, you will see a grid filled with scattered pixels of different colors. 
# To create the output, aggregate all pixels of the same color into a single pixel of that color 
# and place it in the center of the grid. The output grid should be the smallest possible size 
# that can contain the aggregated colors.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Step 1: Find connected components of the same color
    components = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)
    aggregated_components = []

    # Step 2: For each component, find its bounding box and store its color
    for component in components:
        x, y, w, h = bounding_box(component)
        color = component[0, 0]  # Assume the color of the component is the color of its first pixel
        aggregated_components.append((color, (x, y, w, h)))

    # Step 3: Determine the output grid size
    output_width = 0
    output_height = 0
    for color, (x, y, w, h) in aggregated_components:
        output_width = max(output_width, w)
        output_height = max(output_height, h)

    # Step 4: Create the output grid with a black background
    output_grid = np.full((output_height, output_width), Color.BLACK)

    # Step 5: Place the aggregated colors in the output grid
    for color, (x, y, w, h) in aggregated_components:
        sprite = crop(component)
        blit_sprite(output_grid, sprite, x, y, background=Color.BLACK)

    return output_grid