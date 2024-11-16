from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# growth, pixel expansion

# description:
# In the input, you will see a small cluster of colored pixels surrounded by a black background. 
# To create the output, expand the cluster outward by one pixel in all directions, 
# filling the new pixels with the same color as the original cluster.

def transform(input_grid):
    # Create the output grid, starting with the input grid
    output_grid = np.copy(input_grid)

    # Find the background color (assumed to be black)
    background = Color.BLACK

    # Find the connected components in the input grid
    objects = find_connected_components(input_grid, background=background, connectivity=4, monochromatic=True)

    # We expect exactly one object
    assert len(objects) == 1
    original_cluster = objects[0]

    # Get the color of the original cluster
    cluster_color = np.unique(original_cluster[original_cluster!= background])[0]

    # Get the bounding box of the original cluster
    x, y, width, height = bounding_box(original_cluster)

    # Expand the cluster outward by one pixel in all directions
    for i in range(x - 1, x + width + 1):
        for j in range(y - 1, y + height + 1):
            if (i >= 0 and i < output_grid.shape[0] and
                j >= 0 and j < output_grid.shape[1]):
                if output_grid[i, j] == background:
                    output_grid[i, j] = cluster_color

    return output_grid