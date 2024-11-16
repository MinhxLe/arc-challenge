from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color counting, pixel transformation, grid manipulation

# description:
# In the input, you will see a grid filled with colored pixels on a black background.
# To create the output grid, count the number of pixels of each color in the grid.
# Then, replace each pixel in the output grid with the corresponding color based on the following mapping:
# - If a pixel is red, replace it with 3 pixels of red.
# - If a pixel is blue, replace it with 2 pixels of blue.
# - If a pixel is green, replace it with 1 pixel of green.
# - If a pixel is yellow, replace it with 1 pixel of yellow.
# - If a pixel is orange, replace it with 1 pixel of orange.
# - If a pixel is gray, replace it with 1 pixel of gray.
# - If a pixel is pink, replace it with 1 pixel of pink.
# - If a pixel is purple, replace it with 1 pixel of purple.
# - If a pixel is brown, replace it with 1 pixel of brown.
# - If a pixel is black, it remains unchanged.
# If the pixel is any other color, it will be replaced with a pixel of black.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Initialize output grid
    output_grid = np.zeros_like(input_grid)

    # Create color mapping for transformation
    color_mapping = {
        Color.RED: np.array([[Color.RED, Color.RED, Color.RED], [Color.RED, Color.RED, Color.RED]]),
        Color.BLUE: np.array([[Color.BLUE, Color.BLUE], [Color.BLUE, Color.BLUE]]),
        Color.GREEN: np.array([[Color.GREEN, Color.GREEN]]),
        Color.YELLOW: np.array([[Color.YELLOW, Color.YELLOW]]),
        Color.ORANGE: np.array([[Color.ORANGE]]),
        Color.GRAY: np.array([[Color.GRAY]]),
        Color.PINK: np.array([[Color.PINK]]),
        Color.PURPLE: np.array([[Color.PURPLE]]),
        Color.BROWN: np.array([[Color.BROWN]]),
        Color.BLACK: np.array([[Color.BLACK]]),
    }

    # Iterate through the grid and apply the transformation
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            current_color = input_grid[x, y]
            if current_color in color_mapping:
                # Get the color mapping for the current color
                mapping = color_mapping[current_color]
                # Blit the mapped color onto the output grid
                for dx in range(mapping.shape[0]):
                    for dy in range(mapping.shape[1]):
                        output_grid[x + dx, y + dy] = mapping[dx, dy]

    return output_grid