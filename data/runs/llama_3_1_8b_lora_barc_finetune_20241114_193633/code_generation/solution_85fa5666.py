from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, pixel manipulation

# description:
# In the input, you will see a grid containing a pattern of colored pixels, with some regions occluded by black pixels. 
# To create the output, you need to reveal the hidden parts of the pattern by blending the occluded regions with the colors of the visible pixels using a simple averaging method.

def transform(input_grid):
    # Plan:
    # 1. Identify the occluded areas (black pixels)
    # 2. For each occluded pixel, find the colors of the neighboring pixels
    # 3. Blend the colors of the neighboring pixels to fill in the occluded pixel

    # Create an output grid initialized to the input grid
    output_grid = np.copy(input_grid)

    # Identify the occluded areas (black pixels)
    occluded_mask = (input_grid == Color.BLACK)

    # Get the coordinates of the occluded pixels
    occluded_coords = np.argwhere(occluded_mask)

    # For each occluded pixel, find the neighboring pixels
    for occluded_x, occluded_y in occluded_coords:
        # Define the neighborhood (8-way connectivity)
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # skip the occluded pixel itself
                nx, ny = occluded_x + dx, occluded_y + dy
                if 0 <= nx < input_grid.shape[0] and 0 <= ny < input_grid.shape[1]:
                    neighbor_color = input_grid[nx, ny]
                    if neighbor_color!= Color.BLACK:
                        neighbors.append(neighbor_color)

        # If we have neighbors, blend their colors
        if neighbors:
            # Average the colors of the neighboring pixels
            color_count = len(neighbors)
            color_sum = np.zeros(3, dtype=int)  # Assuming colors are represented in RGB
            for color in neighbors:
                color_sum += np.array([1, 0, 0]) if color == Color.RED else np.array([0, 0, 0]) if color == Color.BLACK else np.random.randint(0, 9, size=3)  # Random color for other colors
            average_color = (color_sum / color_count).astype(int)
            # Fill the occluded pixel with the averaged color
            output_grid[occluded_x, occluded_y] = Color.RED if np.all(average_color == np.array([1, 0, 0])) else Color.BLACK  # Assuming RED is a unique color

    return output_grid