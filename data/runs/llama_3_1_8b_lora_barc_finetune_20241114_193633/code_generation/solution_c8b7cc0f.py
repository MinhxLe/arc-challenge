from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern extraction, color blending

# description:
# In the input, you will see a grid with a central pattern surrounded by a border of a different color.
# To make the output, extract the central pattern and fill it with a uniform color that is the average of the colors in the central pattern.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Crop the grid to remove the border
    central_pattern = crop(input_grid, background=Color.BLACK)

    # Calculate the average color of the central pattern
    unique_colors, counts = np.unique(central_pattern[central_pattern!= Color.BLACK], return_counts=True)
    if len(unique_colors) == 0:
        average_color = Color.BLACK  # If no colors found, default to black
    else:
        average_color = unique_colors[np.argmax(counts)]  # Most frequent color

    # Create the output grid filled with the average color
    output_grid = np.full(central_pattern.shape, average_color)

    # Place the central pattern into the output grid
    blit_sprite(output_grid, central_pattern, x=0, y=0, background=average_color)

    return output_grid