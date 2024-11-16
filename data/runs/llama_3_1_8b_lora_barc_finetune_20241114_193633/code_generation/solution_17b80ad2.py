from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# distance transformation, color gradient

# description:
# In the input, you will see a grid with colored pixels scattered throughout, including one or more colored pixels that act as "source" points.
# To make the output, transform all colored pixels based on their distance from the nearest source pixel.
# Each pixel will be colored with a color that corresponds to its distance from the nearest source pixel, using a gradient from black to the color of the source pixel.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.full(input_grid.shape, Color.BLACK)  # Start with a black grid

    # Get the coordinates of all colored pixels
    colored_pixels = np.argwhere(input_grid!= Color.BLACK)

    # For each colored pixel, calculate its distance to the nearest source pixel
    for x, y in colored_pixels:
        # Create a distance map from this pixel
        distance_map = np.full(input_grid.shape, np.inf)
        distance_map[x, y] = 0

        # Use BFS to fill the distance map
        queue = [(x, y)]
        while queue:
            cx, cy = queue.pop(0)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-way connectivity
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < input_grid.shape[0] and 0 <= ny < input_grid.shape[1]:
                    if distance_map[nx, ny] == np.inf and input_grid[nx, ny]!= Color.BLACK:
                        distance_map[nx, ny] = 1
                        queue.append((nx, ny))

        # Color the pixel based on its distance to the nearest source pixel
        nearest_distance = distance_map[x, y]
        if nearest_distance > 0:
            source_color = input_grid[x, y]
            output_grid[x, y] = source_color  # Color the original source pixel
            # Create a gradient color based on the distance
            gradient_color = interpolate_color(source_color, Color.BLACK, nearest_distance)
            for dx in range(-nearest_distance, nearest_distance + 1):
                for dy in range(-nearest_distance, nearest_distance + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < input_grid.shape[0] and 0 <= ny < input_grid.shape[1]:
                        output_grid[nx, ny] = gradient_color

    return output_grid

def interpolate_color(color1, color2, distance):
    # Simple linear interpolation between two colors
    # Assuming colors are represented as RGB tuples (R, G, B)
    r1, g1, b1 = color_to_rgb(color1)
    r2, g2, b2 = color_to_rgb(color2)
    
    # Calculate the interpolated color
    r = int(r1 + (r2 - r1) * (distance / 10))
    g = int(g1 + (g2 - g1) * (distance / 10))
    b = int(b1 + (b2 - b1) * (distance / 10))
    
    return rgb_to_color(r, g, b)

def color_to_rgb(color):
    # Convert color enum to RGB tuple (R, G, B)
    # This is a placeholder function that you should implement based on your color representation
    # For simplicity, we can return a default color for now
    return (0, 0, 0)  # Placeholder for RGB

def rgb_to_color(r, g, b):
    # Convert RGB tuple back to color enum
    # This is a placeholder function that you should implement based on your color representation
    return Color.BLACK  # Placeholder for color mapping