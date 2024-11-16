from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, gradient filling

# description:
# In the input, you will see a grid with two distinct shapes of different colors, separated by a row of black pixels. 
# To create the output, fill the black row with a gradient that transitions from the color of the first shape to the color of the second shape, 
# while preserving the positions of both shapes.

def transform(input_grid):
    # Create the output grid by copying the input grid
    output_grid = np.copy(input_grid)

    # Identify the colors of the two shapes
    first_color = None
    second_color = None

    # Detect the colors of the shapes
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            if input_grid[x, y]!= Color.BLACK:
                if first_color is None:
                    first_color = input_grid[x, y]
                elif second_color is None:
                    second_color = input_grid[x, y]

    # Create a gradient between the two colors
    gradient = np.linspace(np.array([1, 0]), np.array([0, 1]), num=10)
    gradient_colors = [blend_colors(first_color, second_color, t) for t in gradient]

    # Fill the black row with the gradient
    for t in range(len(gradient_colors)):
        for x in range(input_grid.shape[0]):
            if x == 0 or x == input_grid.shape[0] - 1:
                continue  # Skip the first and last rows
            if t == 0 or t == len(gradient_colors) - 1:
                continue  # Skip the first and last gradient steps
            for y in range(input_grid.shape[1]):
                if output_grid[x, y] == Color.BLACK:
                    output_grid[x, y] = gradient_colors[t]

    return output_grid

def blend_colors(color1, color2, t):
    # Simple linear interpolation between two colors
    # For simplicity, we will assume colors are represented as integers and use basic interpolation.
    # This can be enhanced to use RGB or other color representations if needed.
    if color1 == color2:
        return color1
    return color1 if t < 0.5 else color2