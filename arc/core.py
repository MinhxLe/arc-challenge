import typing as ta
from enum import IntEnum, Enum
import numpy as np

MIN_GRID_WIDTH = 1
MAX_GRID_WIDTH = 30
MIN_GRID_HEIGHT = 1
MAX_GRID_HEIGHT = 30

# Tuple of width and height and each entry is an integer representing the color.
Grid = np.ndarray[ta.Tuple[int, int], np.dtype[np.integer]]


class Color(IntEnum):
    """
    Colors are strings (NOT integers), so you CAN'T do math/arithmetic/indexing/ordering on them.
    """

    BLACK = 0
    BLUE = 1
    RED = 2
    YELLOW = 3
    PURPLE = 4
    ORANGE = 5
    GREEN = 6
    BROWN = 7
    GREY = 8
    PINK = 9

    @classmethod
    @property
    def ALL_COLORS(cls) -> list["Color"]:
        return [c for c in cls]

    @classmethod
    @property
    def NOT_BLACK(cls) -> list["Color"]:
        return [c for c in cls if c != Color.BLACK]


class Concept(str, Enum):
    COLOR_MAPPING = "color mapping"
    REFLECTION = "reflection"
    ROTATION = "rotation"
    SYMMETRY = "symmetry"
    TRANSLATION = "translation"
    SELECTION = "selection"
    INVERSION = "inversion"
    RADIATION = "radiation"
    UPSCALING = "upscaling"
    DOWNSCALING = "downscaling"
    GROWTH = "growth"
    SHRINKAGE = "shrinkage"
    BOUNDING_BOX = "bounding box"
    INSERTION = "insertion"
    JIGSAW = "jigsaw"
    RAY_TRACING = "ray tracing"
    BOUNCING = "bouncing"
    FILLING = "filling"
    OCCLUSION = "occlusion"
    REPETITION = "repetition"
    GROUPING = "grouping"
    COUNTING = "counting"
    SWAPPING = "swapping"
    OBJECT_IDENTIFICATION = "object identification"
    OBJECT_SELECTION = "object selection"
    SUBGRID_SELECTION = "subgrid selection"
    ACCRETION = "accretion"
    ABLATION = "ablation"
    SHEARING = "shearing"
    SKEWING = "skewing"
    WARPING = "warping"
    MORPHING = "morphing"
    BLURRING = "blurring"
    SHARPENING = "sharpening"
    MASKING = "masking"
    COMPOSITING = "compositing"
    FILTERING = "filtering"
    COLORIZATION = "colorization"
    DITHERING = "dithering"
    PIXELATING = "pixelating"
    TILING = "tiling"
    TESSELATION = "tesselation"
    CONVOLUTION = "convolution"
    CIRCUMSCRIPTION = "circumscription"
    INSCRIPTION = "inscription"
    PATH_FINDING = "path finding"
    GRAVITY = "gravity"
    SHAPE_MAPPING = "shape mapping"
    COLLISION = "collision"
    STACKING = "stacking"
    NESTING = "nesting"
    SQUEEZING = "squeezing"
    CENTERING = "centering"
    COUNT_MAPPING = "count mapping"
    STRETCHING = "stretching"
    COMPRESSION = "compression"
    EXPANSION = "expansion"
    FALLING = "falling"
    ATTRACTION = "attraction"
    REPULSION = "repulsion"
    FRACTALITY = "fractality"
    ROTATIONAL_SYMMETRY = "rotational symmetry"
    TRANSLATIONAL_SYMMETRY = "translational symmetry"
    ARITHMETIC = "arithmetic"
    ADDITION = "addition"
    SUBTRACTION = "subtraction"
    FRACTURING = "fracturing"
    CUTTING = "cutting"
    SPLITTING = "splitting"
    FUSION = "fusion"
    ATTACHMENT = "attachment"
    DETACHMENT = "detachment"
    TRUNCATION = "truncation"