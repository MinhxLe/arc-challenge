from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color filling, symmetry detection

# description:
# In the input, you will see a grid with a few colored pixels on a black background.
# To create the output, you should find all the colored pixels, and for each pixel, create
# a circular area of the same color centered at that pixel. The radius of the circle should be
# determined by the distance to the nearest pixel of a different color. If a circle would extend
# beyond the bounds of the grid, it should be clipped to fit within the grid.

def transform(input_grid):
    # Create a copy of the input grid to work on
    output_grid = np.copy(input_grid)
    background = Color.BLACK

    # Find the coordinates of all colored pixels
    colored_pixels = np.argwhere(input_grid!= background)

    for x, y in colored_pixels:
        # Get the color of the current pixel
        color = input_grid[x, y]

        # Find all pixels of different colors in the surrounding area
        different_colors = {}
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue  # skip the current pixel
                nx, ny = x + dx, y + dy
                if 0 <= nx < input_grid.shape[0] and 0 <= ny < input_grid.shape[1]:
                    neighbor_color = input_grid[nx, ny]
                    if neighbor_color!= background and neighbor_color!= color:
                        different_colors[neighbor_color] = None

        # Determine the radius based on the distance to the nearest different color
        nearest_dist = float('inf')
        for color2 in different_colors.keys():
            for dx in range(-1, 2):
                for dy in range(-1, 1):
                    if dx == 0 and dy == 0:
                        continue  # skip the center
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < input_grid.shape[0] and 0 <= ny < input_grid.shape[1]:
                        if input_grid[nx, ny] == color2:
                            continue  # skip same color
                        if 0 <= nx < output_grid.shape[0] and 0 <= ny < output_grid.shape[1]:
                            dist = abs(dx) + abs(dy)  # Manhattan distance
                            if dist < nearest_dist:
                                nearest_dist = dist
                                nearest_color = color2

        # Fill the output grid with circles of the determined color
        for dx in range(-1, 2):
            for dy in range(-1, 1):
                if dx == 0 and dy == 0:
                    continue  # skip the center
                nx, ny = x + dx, y + dy
                if 0 <= nx < output_grid.shape[0] and 0 <= ny < output_grid.shape[1]:
                    if output_grid[nx, ny] == background:
                        output_grid[nx, ny] = color
                    else:
                        # If the pixel is already filled with a different color, do not overwrite
                        continue
                # Draw a circle of the determined radius
                radius = nearest_dist + 1  # 1 extra pixel for the border
                for r in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) <= radius:
                        nx2, ny2 = nx + r * dx, ny + r * dy
                        if 0 <= nx2 < output_grid.shape[0] and 0 <= ny2 < output_grid.shape[1]:
                            output_grid[nx2, ny2] = color

    return output_grid