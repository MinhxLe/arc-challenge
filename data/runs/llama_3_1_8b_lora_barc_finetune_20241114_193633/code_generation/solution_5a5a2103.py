from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, pattern recognition, counting

# description:
# In the input, you will see a grid with colored pixels forming a pattern. 
# The task is to count how many pixels of each color are present in the grid and create an output grid
# where each pixel in the output grid is colored according to the following rules:
# - If the pixel is a primary color (red, blue, yellow), it should be replaced by a secondary color:
#   - Red -> Blue, Blue -> Green, Yellow -> Orange.
# - If the pixel is a secondary color, it should be replaced by its corresponding primary color:
#   - Blue -> Red, Green -> Yellow, Orange -> Blue.
# - If the pixel is black, it remains black.
# The output grid should have the same dimensions as the input grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Initialize output grid
    output_grid = input_grid.copy()

    # Create a mapping of primary to secondary colors
    color_map = {
        Color.RED: Color.BLUE,
        Color.BLUE: Color.GREEN,
        Color.YELLOW: Color.ORANGE,
        Color.BLUE: Color.RED,
        Color.GREEN: Color.YELLOW,
        Color.ORANGE: Color.BLUE
    }

    # Apply the color mapping
    output_grid = np.vectorize(lambda color: color_map.get(color, color))(output_grid)

    return output_grid