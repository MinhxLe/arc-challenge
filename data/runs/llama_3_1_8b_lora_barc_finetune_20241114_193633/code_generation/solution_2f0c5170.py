from arc.dsl.common import *
import numpy as np
from typing import *

# concepts:
# object detection, color filtering, cropping

# description:
# In the input, you will see a grid containing various colored objects, some of which are partially occluded by a black rectangle.
# To create the output, you should detect the visible parts of the colored objects that are not covered by the black rectangle
# and crop them out to form a new grid that only contains these visible parts, arranged in the order they appear in the input.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Plan:
    # 1. Detect the black rectangle (occluder)
    # 2. Detect all visible colored objects
    # 3. Crop each visible object to create a new grid

    occlusion_color = Color.BLACK
    black_rectangle_mask = (input_grid == occlusion_color)

    # Find the black rectangle
    black_rectangle_coords = np.argwhere(black_rectangle_mask)
    if black_rectangle_coords.size == 0:
        raise ValueError("No occlusion color found in the grid.")

    # Find the bounding box of the occlusion
    min_x, max_x = black_rectangle_coords[:, 0].min(), black_rectangle_coords[:, 0].max()
    min_y, max_y = black_rectangle_coords[:, 1].min(), black_rectangle_coords[:, 1].max()

    # Detect visible objects (excluding the occluded area)
    visible_objects = detect_objects(grid=input_grid, background=occlusion_color, connectivity=4)

    # Crop visible parts of the objects
    cropped_sprits = []
    for obj in visible_objects:
        cropped_sprint = crop(obj, background=occlusion_color)
        if cropped_sprint.size > 0:  # Ensure it's not empty
            cropped_sprints.append(cropped_sprint)

    # Combine all cropped sprites into a single output grid
    if cropped_sprints:
        output_grid = np.concatenate(cropped_sprints, axis=0)  # Stack vertically
    else:
        output_grid = np.full((1, 1), occlusion_color)  # Default to occlusion color if no visible objects

    return output_grid