from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# shape transformation, color mapping

# description:
# In the input, you will see a grid containing a shape made of colored pixels surrounded by a border of black pixels.
# To make the output, you should mirror the shape horizontally and fill the mirrored area with the same color as the original shape.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Detect the shape by finding the connected components
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)
    assert len(objects) == 1  # There should be exactly one shape in the input

    # Crop the shape to get the sprite
    shape = crop(objects[0], background=Color.BLACK)

    # Get the color of the shape
    shape_color = np.unique(shape[shape!= Color.BLACK])[0]

    # Create an output grid of the same size as the input grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Get the dimensions of the shape
    shape_height, shape_width = shape.shape

    # Mirror the shape horizontally
    mirrored_shape = np.flip(shape, axis=1)

    # Place the original shape and the mirrored shape into the output grid
    for i in range(shape_height):
        for j in range(shape_width):
            if shape[i, j]!= Color.BLACK:
                output_grid[i, j] = shape[i, j]  # Original
                output_grid[i, j + shape_width] = shape[i, j]  # Mirrored

    return output_grid