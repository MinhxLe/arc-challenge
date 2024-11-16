from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, shape transformation

# description:
# In the input, you will see a grid with a shape in the center made up of colored pixels. 
# To create the output, you need to transform the shape by changing the color of each pixel based on its distance from the center of the shape.
# The color mapping is as follows:
# - If a pixel is in the center of the shape, it stays the same color.
# - If a pixel is one pixel away from the center, it changes to the next color in a predefined sequence of colors.
# - If a pixel is two pixels away, it changes to the color after the next in the sequence.
# The output grid will have the same dimensions as the input grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)
    center_x, center_y = input_grid.shape[0] // 2, input_grid.shape[1] // 2
    
    # Define the color sequence
    color_sequence = [
        Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW,
        Color.ORANGE, Color.PINK, Color.PURPLE, Color.BROWN,
        Color.GRAY, Color.GREY, Color.BLACK
    ]
    
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            if input_grid[x, y]!= Color.BLACK:  # Only transform colored pixels
                # Calculate distance from the center
                distance = max(abs(center_x - x), abs(center_y - y))
                # Determine the new color based on distance
                if distance == 0:
                    new_color = input_grid[x, y]
                else:
                    new_color = color_sequence[(distance - 1) % len(color_sequence)]
                output_grid[x, y] = new_color
    
    return output_grid