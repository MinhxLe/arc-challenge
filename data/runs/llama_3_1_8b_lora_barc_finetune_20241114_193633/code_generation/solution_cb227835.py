from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# proximity, distance measurement, color gradient

# description:
# In the input, you will see a grid with a single teal pixel in the center and various other colored pixels surrounding it. 
# To make the output, change the color of each surrounding pixel based on its distance from the teal pixel: 
# closer pixels should become lighter, while farther pixels become darker. 
# The color gradient should transition from teal to black as the distance increases.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)
    center_x, center_y = np.argwhere(input_grid == Color.RED)[0]  # Find the teal pixel's location

    # Define the color gradient based on distance
    max_distance = max(center_x**2 + center_y**2)  # maximum possible distance
    color_gradient = np.linspace(0, 1, num=10)  # 10 steps from teal to black

    # Compute distances for all pixels
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            if output_grid[x, y] == Color.BLACK:  # Only change non-background pixels
                distance = int(np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2))
                # Normalize distance to the gradient
                if distance < max_distance:
                    # Interpolate color based on distance
                    distance_index = int(distance / max_distance * (len(color_gradient) - 1))
                    output_grid[x, y] = Color.RED if distance_index == 0 else Color.BLACK  # Teal to black
                else:
                    output_grid[x, y] = Color.BLACK

    return output_grid