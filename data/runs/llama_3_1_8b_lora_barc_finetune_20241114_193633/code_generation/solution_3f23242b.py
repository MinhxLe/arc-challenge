from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# distance measurement, color mapping, color transformation

# description:
# In the input, you will see a grid with a central green pixel and several colored pixels scattered around.
# To make the output, calculate the distance of each colored pixel from the central green pixel
# and color them based on their distance: 
# closer pixels become lighter shades of green, while farther pixels become darker shades of blue.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    # Find the position of the green pixel
    green_pixel_locations = np.argwhere(input_grid == Color.GREEN)
    assert len(green_pixel_locations) == 1
    center_x, center_y = green_pixel_locations[0]

    # Create a mapping of distances to colors
    max_distance = max(output_grid.shape)  # Maximum possible distance
    color_mapping = {
        0: Color.GREEN,  # Closer distance to the center
        1: Color.GRAY,  # Moderate distance
        2: Color.RED,   # Further distance
        3: Color.BLUE,  # Even further
        4: Color.BLACK, # Farthest distance
        5: Color.BLACK, # Beyond the grid
    }

    # Iterate through each pixel in the grid
    for x in range(output_grid.shape[0]):
        for y in range(output_grid.shape[1]):
            if output_grid[x, y]!= Color.BLACK:  # Ignore the background
                # Calculate the Manhattan distance from the center
                distance = abs(center_x - x) + abs(center_y - y)

                # Map distance to color
                if distance in color_mapping:
                    output_grid[x, y] = color_mapping[distance]
                else:
                    # If distance exceeds the mapping, use black
                    output_grid[x, y] = Color.BLACK

    return output_grid