from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# distance mapping, color gradient

# description:
# In the input, you will see a grid with a colored object on a black background.
# To create the output, compute the distance of each pixel from the center of the object,
# and fill the surrounding pixels with a gradient of colors based on their distance from the center.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find the connected components in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)
    
    # Assuming there's only one object, get the first one
    object_grid = objects[0]
    
    # Get the bounding box of the object
    x, y, width, height = bounding_box(object_grid)
    
    # Create an output grid with the same size as the input grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Get the center of the object
    center_x, center_y = x + width // 2, y + height // 2

    # Fill the output grid based on distance from the center
    for i in range(output_grid.shape[0]):
        for j in range(output_grid.shape[1]):
            # Calculate the Euclidean distance from the center of the object
            distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            # Normalize the distance
            distance_normalized = distance / max(center_x, center_y, output_grid.shape[0] - center_x - 1, output_grid.shape[1] - center_y - 1)
            # Assign colors based on distance
            if distance_normalized < 1.0:
                output_grid[i, j] = Color.RED
            elif distance_normalized < 2.0:
                output_grid[i, j] = Color.GREEN
            elif distance_normalized < 3.0:
                output_grid[i, j] = Color.BLUE
            else:
                output_grid[i, j] = Color.BLACK

    # Overlay the original object onto the output grid
    output_grid = np.where(object_grid!= Color.BLACK, output_grid, Color.BLACK)

    return output_grid