from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# pattern extraction, color replacement, symmetry

# description:
# In the input, you will see a grid filled with various colored pixels. The grid contains a pattern of colored pixels surrounded by black pixels.
# The goal is to extract the colored pattern and replace it with a new color based on the average color of the surrounding black pixels.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create a copy of the input grid to work on
    output_grid = np.copy(input_grid)

    # Detect the connected components (non-black pixels)
    components = find_connected_components(output_grid, background=Color.BLACK, monochromatic=False, connectivity=4)
    
    # For each connected component, we will process it
    for component in components:
        # Get the bounding box of the component
        x, y, width, height = bounding_box(component)

        # Crop the component to isolate the pattern
        pattern = crop(component, background=Color.BLACK)

        # Calculate the average color of the surrounding black pixels
        surrounding_colors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip the center pixel
                nx, ny = x + dx, y + dy
                if 0 <= nx < output_grid.shape[0] and 0 <= ny < output_grid.shape[1]:
                    color = output_grid[nx, ny]
                    if color!= Color.BLACK:
                        surrounding_colors.append(color)

        # Calculate the average color (ignoring black)
        average_color = np.mean([Color.ALL_COLORS if color not in surrounding_colors else Color.BLACK for color in surrounding_colors])
        average_color = np.round(average_color).astype(int)

        # Replace the pattern with the average color
        blit_sprite(output_grid, np.where(pattern!= Color.BLACK, average_color, Color.BLACK), x, y)

    return output_grid