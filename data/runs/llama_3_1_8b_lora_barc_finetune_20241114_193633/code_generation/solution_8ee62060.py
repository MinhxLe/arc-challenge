from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# pattern extraction, mirroring, color swapping

# description:
# In the input you will see a grid with a specific pattern of colored pixels, where the pattern consists of two colors (e.g., blue and red).
# To make the output, mirror the pattern horizontally, and swap the colors of the mirrored pattern with the original pattern.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Extract the original pattern
    original_pattern = crop(input_grid, background=Color.BLACK)

    # Mirror the pattern horizontally
    mirrored_pattern = np.flipud(original_pattern)

    # Create an output grid of the same shape as the input grid
    output_grid = np.copy(input_grid)

    # Swap colors between original and mirrored patterns
    for i in range(original_pattern.shape[0]):
        for j in range(original_pattern.shape[1]):
            if original_pattern[i, j]!= Color.BLACK and mirrored_pattern[i, j]!= Color.BLACK:
                # Swap colors
                output_grid[i, j] = mirrored_pattern[i, j]
                output_grid[i, j + original_pattern.shape[1]] = original_pattern[i, j]

    return output_grid