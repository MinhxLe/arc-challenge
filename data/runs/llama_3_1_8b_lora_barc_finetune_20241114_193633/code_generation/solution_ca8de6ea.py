from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# rotation, color mapping, diagonal symmetry

# description:
# In the input, you will see a grid with a colored pattern that exhibits diagonal symmetry. 
# To make the output, rotate the pattern 90 degrees clockwise and map the colors according to a specified mapping.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Rotate the input grid 90 degrees clockwise
    rotated_grid = np.rot90(input_grid, k=-1)

    # Create a color mapping (for simplicity, we'll just swap colors)
    color_mapping = {
        Color.RED: Color.BLUE,
        Color.BLUE: Color.GREEN,
        Color.GREEN: Color.YELLOW,
        Color.YELLOW: Color.RED,
        Color.GRAY: Color.PINK,
        Color.PINK: Color.PURPLE,
        Color.PURPLE: Color.BROWN,
        Color.BROWN: Color.ORANGE,
        Color.ORANGE: Color.GRAY,
        Color.BLACK: Color.BLACK  # Keep background unchanged
    }

    # Map colors in the rotated grid according to the mapping
    output_grid = np.vectorize(lambda color: color_mapping.get(color, color))(rotated_grid)

    return output_grid