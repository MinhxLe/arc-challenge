from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, gradient filling, proximity

# description:
# In the input, you will see a grid with a pattern of colored pixels and a black background. 
# To create the output, fill in the black areas of the grid with colors that correspond to their proximity 
# to the nearest colored pixel in the pattern, creating a gradient effect.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid for the output
    output_grid = np.copy(input_grid)

    # Get the dimensions of the grid
    height, width = output_grid.shape

    # Iterate through each pixel in the grid
    for x in range(height):
        for y in range(width):
            # If the current pixel is black, we need to fill it
            if output_grid[x, y] == Color.BLACK:
                # Find the nearest colored pixel
                nearest_color = None
                min_distance = float('inf')
                
                for i in range(height):
                    for j in range(width):
                        if output_grid[i, j]!= Color.BLACK:
                            # Calculate the Euclidean distance
                            distance = np.sqrt((x - i) ** 2 + (y - j) ** 2)
                            if distance < min_distance:
                                min_distance = distance
                                nearest_color = output_grid[i, j]

                # Fill the current pixel with the nearest color
                output_grid[x, y] = nearest_color

    return output_grid