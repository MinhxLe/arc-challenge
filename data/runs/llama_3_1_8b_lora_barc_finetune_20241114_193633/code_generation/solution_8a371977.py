from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color gradient, object detection, masking

# description:
# In the input, you will see a grid filled with various colored objects. To make the output, apply a color gradient effect to each object based on its distance from the center of the grid, 
# while maintaining the shapes of the objects. The closer an object is to the center, the more saturated its color will be, while the farther away it is, the more faded it will appear.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Plan:
    # 1. Find all connected components (objects) in the input grid.
    # 2. Calculate the center of the grid.
    # 3. For each object, calculate its distance from the center.
    # 4. Create a color gradient effect based on the distance, and apply it to the object in the output grid.

    output_grid = np.copy(input_grid)

    # Step 1: Find connected components (objects) in the grid
    objects = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True, connectivity=8)

    # Step 2: Get the dimensions of the grid
    center_x, center_y = input_grid.shape[0] // 2, input_grid.shape[1] // 2

    for obj in objects:
        # Step 3: Calculate the bounding box of the object
        x, y, width, height = bounding_box(obj, background=Color.BLACK)
        # Step 4: Create a mask for the object
        mask = np.zeros_like(input_grid)
        blit_sprite(mask, obj, x=x, y=y, background=Color.BLACK)

        # Calculate the distance from the center for each pixel in the object
        for i in range(height):
            for j in range(width):
                if mask[i, j]!= Color.BLACK:  # Only consider non-background pixels
                    distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                    # Normalize the distance to a value between 0 and 1
                    normalized_distance = distance / (max(center_x, center_y))
                    # Calculate the color intensity (saturation) based on the distance
                    saturation = max(0, 1 - normalized_distance)
                    color = input_grid[x + i, y + j]  # Get the color of the current pixel
                    if color!= Color.BLACK:  # Only change color if it's not background
                        # Apply a simple color gradient effect: fading effect
                        output_grid[x + i, y + j] = color if saturation > 0.5 else Color.BLACK

    return output_grid