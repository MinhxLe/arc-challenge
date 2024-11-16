from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, pixel replacement, symmetry

# description:
# In the input, you will see a grid containing various colored shapes (objects) on a black background.
# To make the output grid, replace the color of each object with a new color according to a specified mapping:
# red -> blue, green -> yellow, blue -> red, yellow -> green, pink -> orange, orange -> pink.
# The shapes should maintain their positions but change their colors based on the mapping.

def transform(input_grid):
    # Initialize output grid
    output_grid = input_grid.copy()

    # Perform color mapping
    output_grid = np.vectorize(lambda color: color_map.get(color, color))(output_grid)

    return output_grid

# Constructing the color map
color_map = {
    Color.RED: Color.BLUE,
    Color.GREEN: Color.YELLOW,
    Color.BLUE: Color.RED,
    Color.YELLOW: Color.GREEN,
    Color.PINK: Color.ORANGE,
    Color.ORANGE: Color.PINK
}