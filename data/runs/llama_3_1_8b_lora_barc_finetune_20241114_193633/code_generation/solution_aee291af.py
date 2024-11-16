from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# color transformation, bounding box extraction, symmetry detection

# description:
# In the input, you will see a grid filled with colored pixels on a black background.
# The goal is to identify the largest connected component of a specific color, and extract it.
# The output should be a grid that contains only the largest connected component of the specified color, 
# centered in a new grid with a black background. If no component of the specified color is found, 
# return an empty grid.

def transform(input_grid: np.ndarray, color: str = Color.RED) -> np.ndarray:
    # Find connected components in the grid
    components = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=True)

    # Identify the largest component of the specified color
    largest_component = max(components, key=lambda obj: np.sum(obj == color), default=None)

    # If no component found, return a black grid
    if largest_component is None:
        return np.zeros_like(input_grid)

    # Crop the largest component to get the sprite
    sprite = crop(largest_component, background=Color.BLACK)

    # Create an output grid large enough to fit the sprite centered
    output_grid_size = (sprite.shape[0] + 2, sprite.shape[1] + 2)  # Add padding
    output_grid = np.full(output_grid_size, Color.BLACK)

    # Center the sprite in the output grid
    start_x = 1  # Centering by 1 pixel
    start_y = 1  # Centering by 1 pixel
    blit_sprite(output_grid, sprite, x=start_x, y=start_y, background=Color.BLACK)

    return output_grid