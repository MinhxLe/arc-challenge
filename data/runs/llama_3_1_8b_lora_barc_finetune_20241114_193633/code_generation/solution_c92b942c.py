from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# symmetry detection, pattern replication

# description:
# In the input, you will see a grid with a central pattern that has been partially occluded by black pixels. 
# The output should replicate the visible parts of the pattern symmetrically in all four quadrants of the grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Plan:
    # 1. Find the non-black pixels to identify the visible pattern.
    # 2. Detect the center of the pattern.
    # 3. Create the output grid which is larger to accommodate the symmetrical replication.
    # 4. Copy the pattern into all four quadrants of the output grid.

    # Step 1: Find the non-black pixels
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)
    assert len(objects) > 0, "No pattern found in the input grid"

    # Assuming the first object is the main pattern
    pattern = objects[0]
    pattern_bbox = bounding_box(pattern)

    # Step 2: Calculate the center of the pattern
    center_x = pattern_bbox[0] + pattern.shape[0] // 2
    center_y = pattern_bbox[1] + pattern.shape[1] // 2

    # Step 3: Create the output grid
    output_size = (pattern.shape[0] * 2, pattern.shape[1] * 2)
    output_grid = np.full(output_size, Color.BLACK)

    # Step 4: Blit the pattern into all four quadrants
    for dx in [0, 1]:  # 0 for left, 1 for right
        for dy in [0, 1]:  # 0 for top, 1 for bottom
            blit_sprite(output_grid, pattern, x=center_x + dx * pattern.shape[0] // 2, y=center_y + dy * pattern.shape[1] // 2, background=Color.BLACK)

    return output_grid