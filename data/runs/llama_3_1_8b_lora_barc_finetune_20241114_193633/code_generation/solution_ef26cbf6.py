from arc.dsl.common import *

import numpy as np
from typing import *

# concepts:
# color mapping, symmetry detection, transformation

# description:
# In the input, you will see a grid with a black background and a colored shape that has a specific color pattern.
# To create the output, you need to detect the symmetrical properties of the shape and replace the original color with a new color based on its symmetry.
# The new color will be determined based on the detected symmetry type:
# 1. If the shape is horizontally symmetrical, change its color to green.
# 2. If it is vertically symmetrical, change its color to blue.
# 3. If it is diagonally symmetrical, change its color to yellow.
# If it is not symmetrical, leave the color unchanged.

def transform(input_grid):
    # Detect translational symmetries in the input grid
    symmetries = detect_translational_symmetry(input_grid, ignore_colors=[Color.BLACK])
    
    # Initialize the output grid with the same shape as input
    output_grid = np.copy(input_grid)
    
    # Check for horizontal symmetry
    horizontal_symmetry = detect_mirror_symmetry(input_grid, background=Color.BLACK, connectivity=4)
    if horizontal_symmetry:
        output_grid = np.where(input_grid!= Color.BLACK, Color.GREEN, Color.BLACK)

    # Check for vertical symmetry
    vertical_symmetry = detect_mirror_symmetry(input_grid, background=Color.BLACK, connectivity=4)
    if vertical_symmetry:
        output_grid = np.where(input_grid!= Color.BLACK, Color.BLUE, Color.BLACK)

    # Check for diagonal symmetry
    diagonal_symmetry = detect_mirror_symmetry(input_grid, background=Color.BLACK, connectivity=8)
    if diagonal_symmetry:
        output_grid = np.where(input_grid!= Color.BLACK, Color.YELLOW, Color.BLACK)
    
    return output_grid