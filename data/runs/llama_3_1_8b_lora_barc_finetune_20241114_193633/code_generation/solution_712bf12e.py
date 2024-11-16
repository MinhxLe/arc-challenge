from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# distance measurement, proximity, color blending

# description:
# In the input, you will see a grid with several colored pixels scattered throughout. 
# To create the output, measure the distance of each colored pixel to the nearest red pixel and color all pixels that are within a certain distance threshold with a unique color (e.g., red), 
# while leaving the rest unchanged.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create an output grid initialized to the same as input
    output_grid = np.copy(input_grid)
    
    # Get the coordinates of all red pixels
    red_coords = np.argwhere(input_grid == Color.RED)

    # Define a threshold for proximity
    threshold = 3

    for x, y in np.argwhere(input_grid!= Color.BLACK):
        # Calculate the distance to the nearest red pixel
        min_distance = float('inf')
        nearest_red = None
        
        for rx, ry in red_coords:
            distance = np.sqrt((x - rx) ** 2 + (y - ry) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_red = (rx, ry)

        # If there is a red pixel within the threshold, color the pixel red
        if nearest_red is not None and min_distance < threshold:
            output_grid[x, y] = Color.RED

    return output_grid