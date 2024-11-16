from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# color mapping, filling, symmetry

# description:
# In the input, you will see a grid with several colored pixels arranged in a symmetrical pattern. 
# The output should replace the colors in the pattern according to the following mapping:
# blue -> red, green -> yellow, yellow -> blue, and red -> green.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Initialize output grid
    output_grid = np.copy(input_grid)

    # Perform color mapping
    output_grid = np.vectorize(lambda color: color_map.get(color, color))(output_grid)

    return output_grid

# Constructing the color map
color_map = {
    Color.BLUE: Color.RED,
    Color.GREEN: Color.YELLOW,
    Color.YELLOW: Color.BLUE,
    Color.RED: Color.GREEN
}