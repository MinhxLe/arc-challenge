from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# color mapping, grid transformation

# description:
# In the input grid, you will see several colored squares on a black background. 
# To make the output grid, change the color of each square to a new color according to the following mapping:
# red -> teal, blue -> green, yellow -> orange, green -> blue, orange -> yellow, teal -> red, 
# pink -> gray, gray -> pink, purple -> brown, brown -> purple.
# The output grid should be the same size as the input grid.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Initialize output grid
    output_grid = input_grid.copy()

    # Create a color mapping
    color_map = {
        Color.RED: Color.BLUE,
        Color.BLUE: Color.GREEN,
        Color.YELLOW: Color.ORANGE,
        Color.GREEN: Color.BLUE,
        Color.ORANGE: Color.YELLOW,
        Color.RED: Color.BLUE,
        Color.PINK: Color.GRAY,
        Color.GRAY: Color.PINK,
        Color.PURPLE: Color.BROWN,
        Color.BROWN: Color.PURPLE,
    }

    # Perform the color mapping
    output_grid = np.vectorize(lambda color: color_map.get(color, color))(output_grid)

    return output_grid