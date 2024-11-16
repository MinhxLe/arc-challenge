from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# color transformation, reflection, mirroring

# description:
# In the input, you will see a grid with a colored pattern on a black background.
# To create the output, mirror the pattern horizontally and change the color of the mirrored part to a different color.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Detect the bounding box of the non-background pixels
    bounding_box_coords = bounding_box(input_grid, background=Color.BLACK)
    x, y, width, height = bounding_box_coords

    # Crop the pattern from the input grid
    pattern = crop(input_grid, background=Color.BLACK)

    # Create a new output grid with the same size as the input grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Mirror the pattern horizontally
    mirrored_pattern = np.copy(pattern)
    mirrored_pattern = mirrored_pattern[:, ::-1]

    # Change the color of the mirrored part to a different color
    new_color = Color.RED  # You can choose any color that is not in the original pattern
    mirrored_pattern[mirrored_pattern!= Color.BLACK] = new_color

    # Place the original pattern and the mirrored pattern into the output grid
    blit_sprite(output_grid, pattern, x, y, background=Color.BLACK)
    blit_sprite(output_grid, mirrored_pattern, x + width, y, background=Color.BLACK)

    return output_grid