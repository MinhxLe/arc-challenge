from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# color extraction, unique color mapping, grid transformation

# description:
# In the input, you will see a grid with multiple objects of different colors. 
# To make the output, extract the unique colors from the objects and map them to a new color scheme 
# where each unique color corresponds to a new color from a predefined palette.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Step 1: Detect all connected components in the input grid
    objects = detect_objects(input_grid, monochromatic=False, background=Color.BLACK, connectivity=4)

    # Step 2: Extract unique colors from the detected objects
    unique_colors = {}
    for obj in objects:
        for x, y in np.argwhere(obj!= Color.BLACK):
            color = obj[x, y]
            if color not in unique_colors:
                unique_colors[color] = len(unique_colors)

    # Step 3: Create a color mapping from unique colors to a predefined palette
    color_mapping = {Color.BLUE: Color.RED, Color.GREEN: Color.YELLOW, Color.RED: Color.BLUE, 
                     Color.ORANGE: Color.PINK, Color.PURPLE: Color.BROWN, Color.GRAY: Color.GRAY, 
                     Color.BLACK: Color.BLACK, Color.YELLOW: Color.YELLOW, Color.PINK: Color.PURPLE, 
                     Color.BROWN: Color.GREEN}

    # Step 4: Create the output grid and fill it with mapped colors
    output_grid = np.full(input_grid.shape, Color.BLACK)  # Initialize output grid with background color

    for unique_color, new_color in color_mapping.items():
        # Replace unique colors in the output grid with the corresponding new color
        output_grid[output_grid == unique_color] = new_color

    # Step 5: Crop the output grid to remove any extra rows or columns of black pixels
    output_grid = crop(output_grid, background=Color.BLACK)

    return output_grid