from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# color mapping, grid transformation, object placement

# description:
# In the input, you will see a grid with several colored objects and a single blue square at a random location.
# The output should replace each object's color with the color of the blue square, but only if the object's position matches the position of the blue square.
# If there are multiple objects in the same position, they will all take on the color of the blue square.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    # Find the position of the blue square
    blue_square_position = np.argwhere(input_grid == Color.BLUE)
    
    # Get the color of the blue square
    blue_color = Color.BLUE

    # Find all connected components in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=4, monochromatic=False)

    # Replace the colors of the objects in the output grid that match the blue square's position
    for obj in objects:
        obj_color_positions = np.argwhere(obj!= Color.BLACK)
        if np.any(obj_color_positions == blue_square_position[0]):
            # Replace the color of the object with the blue square's color
            obj[obj!= Color.BLACK] = blue_color

    return output_grid