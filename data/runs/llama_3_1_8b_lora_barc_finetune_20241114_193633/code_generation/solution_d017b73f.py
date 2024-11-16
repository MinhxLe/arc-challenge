from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# stacking, color blending, pixel transformation

# description:
# In the input, you will see a grid with colored blocks stacked vertically. 
# Each block has a color, and they may overlap. 
# To create the output grid, stack these blocks on top of each other while blending the colors 
# of overlapping pixels. The resulting color for overlapping pixels is determined by averaging the RGB values of the colors.

def transform(input_grid):
    output_grid = np.full(input_grid.shape, Color.BLACK)  # Start with a black background
    height, width = input_grid.shape

    # Iterate through each column to stack the blocks
    for x in range(width):
        current_height = 0
        for y in range(height):
            if input_grid[y, x]!= Color.BLACK:  # If the pixel is colored
                color = input_grid[y, x]
                # If there are already pixels in the output grid, blend the colors
                if output_grid[current_height, x]!= Color.BLACK:
                    # Average the existing color with the new color
                    existing_color = output_grid[current_height, x]
                    new_color = average_colors(existing_color, color)
                    output_grid[current_height, x] = new_color
                else:
                    # If it's the first pixel, just place it
                    output_grid[current_height, x] = color
                current_height += 1  # Move down for the next block

    return output_grid

def average_colors(color1, color2):
    """
    A simple function to average two colors represented as strings.
    For simplicity, we will define a basic color mapping to RGB values.
    """
    color_map = {
        Color.BLUE: (0, 0, 255),
        Color.RED: (255, 0, 0),
        Color.GREEN: (0, 255, 0),
        Color.YELLOW: (255, 255, 0),
        Color.GRAY: (128, 128, 128),
        Color.PINK: (255, 192, 203),
        Color.ORANGE: (255, 165, 0),
        Color.PURPLE: (0, 128, 128),
        Color.BROWN: (128, 0, 0),
        Color.BLACK: (0, 0, 0),
    }
    
    # Convert colors to RGB
    rgb1 = np.array(color_map[color1])
    rgb2 = np.array(color_map[color2])
    
    # Average the RGB values
    avg_rgb = (rgb1 + rgb2) / 2
    # Find the closest color in the color_map
    closest_color = min(color_map.items(), key=lambda c: np.linalg.norm(np.array(c[1]) - avg_rgb))
    return closest_color[0]  # Return the color name