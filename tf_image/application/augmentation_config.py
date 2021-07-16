from enum import IntEnum


# use int enum because of TensorFlow
class ColorAugmentation(IntEnum):
    """
    Enum which splits color related operations into two groups:
    - LIGHT: brightness, contrast
    - MEDIUM: LIGHT +  hue, saturation
    - AGGRESSIVE: MEDIUM + channel swap, channel drop, gray scale.

    In addition, there is an option for no augmentations: ColorAugmentation.NONE.
    """

    NONE = 0
    LIGHT = 1
    MEDIUM = 2
    AGGRESSIVE = 3


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
        self.quality = True  # jpeg quality, noise
        self.erasing = True
        self.rotate90 = False
        self.rotate45 = False  # rotate 45 degrees clockwise (other multiples can be done by turning on rotate90)
        self.rotate_max = 13
        self.flip_vertical = True
        self.flip_horizontal = True
        self.padding_square = False
