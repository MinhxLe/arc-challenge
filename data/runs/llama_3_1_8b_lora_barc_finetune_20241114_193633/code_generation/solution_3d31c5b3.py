from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# color blending, pattern extraction

# description:
# In the input, you will see a grid with several colored patterns (a black background and colored pixels).
# To make the output, extract the unique patterns from the grid and blend them into a new grid,
# creating a gradient effect from one color to another based on their proximity to each other.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find all connected components (patterns) in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK, connectivity=8)

    # Prepare the output grid
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Dictionary to store the color and its coordinates
    color_positions = {}

    for obj in objects:
        # Get the bounding box of the object
        x, y, width, height = bounding_box(obj, background=Color.BLACK)
        # Crop the object to isolate its color
        cropped_obj = crop(obj, background=Color.BLACK)
        
        # Get the color of the object
        color = cropped_obj[0, 0]  # Assuming the object is monochromatic

        # Store the positions of the color in a list
        if color not in color_positions:
            color_positions[color] = []
        color_positions[color].append((x, y))

    # Create a blended output grid
    for color, positions in color_positions.items():
        for pos_x, pos_y in positions:
            # Create a gradient effect based on the position of the color
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue  # Skip the original position
                    # Calculate the blended color based on distance from the original position
                    distance = max(abs(dx), abs(dy))
                    if distance == 1:  # Closer to the original position
                        output_grid[pos_x + dx, pos_y + dy] = color
                    elif distance == 2:  # Further away, blend to black
                        output_grid[pos_x + dx, pos_y + dy] = Color.BLACK

    return output_grid