from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# color mapping, symmetry detection

# description:
# In the input grid, you will see a pattern of colored pixels arranged in a symmetric way (either horizontal, vertical, or both).
# To create the output, identify the dominant color in the pattern and map it to a new color based on a predefined mapping.
# The output should retain the original pattern's structure while changing its colors according to the mapping.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Define the color mapping
    color_map = {
        Color.RED: Color.GREEN,
        Color.BLUE: Color.YELLOW,
        Color.GREEN: Color.RED,
        Color.YELLOW: Color.BLUE,
        Color.PURPLE: Color.BROWN,
        Color.BROWN: Color.PURPLE,
        Color.GRAY: Color.PINK,
        Color.PINK: Color.GRAY,
        Color.ORANGE: Color.BLACK,
        Color.BLACK: Color.ORANGE,
    }

    # Detect connected components in the input grid
    components = find_connected_components(input_grid, background=Color.BLACK, monochromatic=True)
    
    # Initialize output grid
    output_grid = np.full_like(input_grid, Color.BLACK)

    # For each connected component, determine its color and map it
    for component in components:
        # Get the color of the current component
        color = component[0, 0]  # Assuming the component is monochromatic
        if color in color_map:
            # Map the color
            new_color = color_map[color]
            # Find the bounding box of the component
            x, y, width, height = bounding_box(component)
            # Fill the output grid with the new color
            output_grid[x:x + width, y:y + height] = new_color

    return output_grid