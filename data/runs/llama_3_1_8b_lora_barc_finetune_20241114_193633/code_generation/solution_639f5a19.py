from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# distance measurement, color mapping

# description:
# In the input, you will see a grid with a black background and colored pixels scattered throughout. 
# The task is to color each pixel based on its distance from the nearest colored pixel. 
# The closer a pixel is to a colored pixel, the closer its color should be to the original color of that pixel. 
# The output grid will reflect this distance-based coloring.

def transform(input_grid):
    # Plan:
    # 1. Create an output grid initialized to black
    # 2. For each pixel in the input grid, determine the distance to the nearest colored pixel
    # 3. Color the output pixel based on the distance to the nearest colored pixel

    output_grid = np.full_like(input_grid, Color.BLACK)

    # Get the coordinates of all colored pixels
    colored_pixels = np.argwhere(input_grid!= Color.BLACK)

    for x, y in np.argwhere(input_grid!= Color.BLACK):
        color = input_grid[x, y]
        
        # Calculate the distance from the colored pixel to all other pixels
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                if (i, j) == (x, y):
                    continue  # Skip the original colored pixel
                distance = np.sqrt((i - x) ** 2 + (j - y) ** 2)
                if distance > 0:  # Only consider non-background pixels
                    # Determine the color based on the distance
                    if distance < 1.5:  # Closer distance -> original color
                        output_grid[i, j] = color
                    elif distance < 2.5:  # Medium distance -> a blend of original color
                        output_grid[i, j] = color
                    else:  # Further distance -> black (or background)
                        output_grid[i, j] = Color.BLACK

    return output_grid