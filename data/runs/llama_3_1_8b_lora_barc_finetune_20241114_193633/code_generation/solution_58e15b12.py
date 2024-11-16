from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# radial symmetry, color mapping, pattern generation

# description:
# In the input, you will see a grid with a central colored pixel and several colored pixels surrounding it in a circular pattern.
# To make the output grid, replicate the circular pattern of colored pixels around the central pixel, ensuring that the colors follow a radial symmetry around the center.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Find the center of the input grid
    center_x, center_y = input_grid.shape[0] // 2, input_grid.shape[1] // 2

    # Get the color of the central pixel
    central_color = input_grid[center_x, center_y]

    # Create the output grid, initially filled with the background color
    output_grid = np.full(input_grid.shape, Color.BLACK)

    # Get the surrounding colored pixels around the center
    surrounding_colors = []
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:
                continue  # skip the central pixel
            if (0 <= center_x + dx < input_grid.shape[0]) and (0 <= center_y + dy < input_grid.shape[1]):
                if input_grid[center_x + dx, center_y + dy]!= Color.BLACK:
                    surrounding_colors.append(input_grid[center_x + dx, center_y + dy])

    # Create radial symmetry around the central pixel
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:
                continue  # skip the center
            # Calculate the symmetric positions
            sym_x = center_x + dx
            sym_y = center_y + dy
            # Set the symmetric positions in the output grid
            output_grid[sym_x, sym_y] = surrounding_colors.pop(0)  # Fill with the first surrounding color

            # Create the radial symmetry
            for radius in range(1, max(len(surrounding_colors), 1)):
                # Calculate the surrounding positions
                for angle in range(0, 360, 45):  # 8 symmetrical positions (every 45 degrees)
                    rad = np.radians(angle)
                    sym_x = int(center_x + radius * np.cos(rad))
                    sym_y = int(center_y + radius * np.sin(rad))
                    if 0 <= sym_x < output_grid.shape[0] and 0 <= sym_y < output_grid.shape[1]:
                        output_grid[sym_x, sym_y] = surrounding_colors.pop(0) if surrounding_colors else Color.BLACK

    return output_grid