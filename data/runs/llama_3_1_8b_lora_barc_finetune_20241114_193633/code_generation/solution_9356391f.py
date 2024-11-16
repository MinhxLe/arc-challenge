from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color blending, gradient creation, interpolation

# description:
# In the input, you will see a grid with a central colored object surrounded by a black background.
# To create the output grid, you should blend the color of the central object with the black background,
# creating a gradient effect that transitions from the object's color to black as you move away from the object.

def transform(input_grid: np.ndarray) -> np.ndarray:
    output_grid = np.copy(input_grid)

    # Find the color of the central object
    central_object_color = np.unique(output_grid[output_grid!= Color.BLACK])[0]

    # Create a gradient effect towards black
    height, width = output_grid.shape
    for x in range(height):
        for y in range(width):
            # Calculate the distance from the object
            distance = max(abs(x - (height // 2)), abs(y - (width // 2))
            # Blend color based on the distance
            if output_grid[x, y] == central_object_color:
                # If it's part of the object, keep the color
                output_grid[x, y] = central_object_color
            else:
                # Blend towards black based on distance
                blend_factor = max(0, 1 - (distance / max(height, width) * 0.5))  # Normalize distance to [0, 1]
                output_grid[x, y] = blend_color(central_object_color, Color.BLACK, blend_factor)

    return output_grid


def blend_color(color1, color2, blend_factor):
    """Blend two colors together based on the blend factor."""
    # Simple linear interpolation
    return color1 if blend_factor > 0.5 else color2