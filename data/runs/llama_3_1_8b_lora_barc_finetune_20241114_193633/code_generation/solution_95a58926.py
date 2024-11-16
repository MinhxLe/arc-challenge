from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel clustering, color averaging, grid transformation

# description:
# In the input, you will see a grid filled with colored pixels. The task is to identify clusters of connected pixels of the same color 
# and replace each cluster with a single pixel of a new color that is the average of the original cluster's color.
# The output grid should have the same dimensions as the input grid.

def transform(input_grid):
    # Create an output grid initialized to the background color
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Find connected components in the grid
    components = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)

    for component in components:
        # Calculate the average color of the component
        unique_colors, counts = np.unique(component, return_counts=True)
        color_counts = dict(zip(unique_colors, counts))
        average_color = max(color_counts, key=color_counts.get)

        # Get the bounding box of the component
        x, y, w, h = bounding_box(component)

        # Fill the output grid with the average color at the position of the original component
        output_grid[x:x+w, y:y+h] = average_color

    return output_grid