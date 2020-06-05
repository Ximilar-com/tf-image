from enum import IntEnum


# use int enum because of TensorFlow
class ColorAugmentation(IntEnum):
    """
    Enum which splits color related operations into two groups:
    - MILD: brightness, contrast, hue, saturation and
    - AGGRESSIVE: all from MILD and chanel swap, channel drop, gray scale.

    In addition, there is an option for no augmentations: ColorAugmentation.NONE.
    """

    NONE = 0
    MILD = 1
    AGGRESSIVE = 2


class AspectRatioAugmentation(IntEnum):
    """
    There are two posibilities how we can distort aspect ration:
    - NORMAL: same maximal distortions in both horizontal and vertical direction or
    - TOWARDS_SQUARE: more squeezing for longer side and more stretching for a shorter side.

    In addition, there is an option for no augmentations: AspectRatioAugmentation.NONE.
    """

    NONE = 0
    NORMAL = 1
    TOWARDS_SQUARE = 2


class AugmentationConfig(object):
    """
    Specifies which augmentations should be applied.
    """

    def __init__(self):
        self.color = ColorAugmentation.AGGRESSIVE
        self.crop = True
        self.distort_aspect_ratio = AspectRatioAugmentation.NORMAL
        self.quality = True  # # jpeg quality, noise
        self.erasing = True
        self.rotate90 = False
        self.rotate_max = 13
        self.flip_vertical = True
        self.flip_horizontal = True
        self.padding_square = False
