from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# object extraction, color mapping, transformation

# description:
# In the input, you will see a grid with a central object surrounded by several colored pixels.
# To make the output, extract the central object, and then replace its color with the average color
# of the surrounding pixels.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create an output grid initialized to the background color
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Extract the central object by cropping the grid to the bounding box of non-background pixels
    central_object = crop(input_grid, background=Color.BLACK)

    # Calculate the bounding box of the central object
    x, y, width, height = bounding_box(central_object)

    # Extract the surrounding pixels
    surrounding_pixels = np.concatenate((
        input_grid[max(0, x - 1):x + width + 1, max(0, y - 1):y + height + 1],
        input_grid[max(0, x - 1):x + width + 1, max(0, y - 1):y + height + 1]
    ))

    # Filter out the background color
    surrounding_colors = surrounding_pixels[surrounding_pixels!= Color.BLACK]

    # Calculate the average color of the surrounding pixels
    if surrounding_colors.size > 0:
        average_color = np.mean(surrounding_colors)
        average_color = round(average_color)  # Round to nearest integer
    else:
        average_color = Color.BLACK  # If no surrounding colors, default to black

    # Change the color of the central object to the average color
    output_grid[x:x + central_object.shape[0], y:y + central_object.shape[1]] = central_object
    output_grid[output_grid!= Color.BLACK] = average_color

    return output_grid