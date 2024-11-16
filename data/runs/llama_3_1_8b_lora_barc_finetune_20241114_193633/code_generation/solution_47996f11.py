from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pixel clustering, color mapping, counting

# description:
# In the input, you will see a grid filled with various colored pixels. The goal is to transform the grid by counting the number of contiguous pixels of each color, 
# and then filling the output grid with colors based on these counts. Each unique color will be represented by a specific color in the output grid.
# The output grid should be filled with the colors in a specific order based on the count of pixels in the input grid.

def transform(input_grid):
    # Create a dictionary to count occurrences of each color
    color_counts = {}
    
    # Count the pixels
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            color = input_grid[x, y]
            if color!= Color.BLACK:  # Ignore background color
                if color in color_counts:
                    color_counts[color] += 1
                else:
                    color_counts[color] = 1

    # Sort colors by their counts (descending)
    sorted_colors = sorted(color_counts.items(), key=lambda item: item[1], reverse=True)

    # Create output grid with the same shape as input
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Fill the output grid with colors based on sorted counts
    color_index = 0
    for color, count in sorted_colors:
        # Fill the output grid with the color
        for i in range(count):
            x, y = random_free_location_for_sprite(output_grid, background=Color.BLACK)
            output_grid[x, y] = color

        # Move to the next color
        color_index += 1
        if color_index < len(sorted_colors):
            color_index += 1  # Skip the next color for the next row

    return output_grid