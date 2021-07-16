import tensorflow as tf

from tf_image.application.augmentation_config import ColorAugmentation, AugmentationConfig, AspectRatioAugmentation
from tf_image.core.bboxes.clip import clip_random_with_bboxes
from tf_image.core.bboxes.erase import multiple_erase, calculate_bboxes_max_erase_area
from tf_image.core.bboxes.flip import flip_left_right, flip_up_down
from tf_image.core.bboxes.resize import random_aspect_ratio_deformation, random_pad_to_square
from tf_image.core.bboxes.rotate import random_rotate, rot90, rot45
from tf_image.core.clip import clip_random
from tf_image.core.colors import channel_drop, grayscale, channel_swap, rgb_shift
from tf_image.core.convert_type_decorator import convert_type
from tf_image.core.quality import gaussian_noise
from tf_image.core.random import random_function
from tf_image.core.random import random_function_bboxes


def random_augmentations(image, augmentation_config: AugmentationConfig, bboxes=None, prob_demanding_ops: float = 0.5):
    """
    Apply augmentations in random order.

    WARNING: this is just a testing class and it is likely to change.

    :param image: 3-D Tensor of shape (height, width, channels).
    :param augmentation_config: Config defining which augmentations can be applied.
    :param bboxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [ymin, xmin, ymin, xmax]
    :param prob_demanding_ops: Probability that a time consuming operation (like rotation) will be performed.
    :return: augmented image or (augmented image, bboxes) if bboxes parameter is not None
    """
    has_bboxes = bboxes is not None
    if not has_bboxes:
        bboxes = tf.reshape([], (0, 4))

    # convert_dtype decorator needs this special argument order (converting now saves us converting in each operation)
    image, bboxes = _random_augmentations(image, bboxes, augmentation_config, prob_demanding_ops)

    if has_bboxes:
        return image, bboxes

    return image


@tf.function
@convert_type
def _random_augmentations(image, bboxes, augmentation_config: AugmentationConfig, prob_demanding_ops: float):
    @tf.function
    def apply(idx, image, bboxes):
        # List of tuples (precondition, augmentation), augmentation will be applied only if precondition is True.
        functions = [
            (
                tf.math.equal(augmentation_config.color, ColorAugmentation.AGGRESSIVE),
                lambda: (
                    (
                        random_function(image, rgb_shift, 0.2, **{"r_shift": 0.15, "g_shift": 0.15, "b_shift": 0.15}),
                        bboxes,
                    )
                ),
            ),
            (
                tf.math.equal(augmentation_config.color, ColorAugmentation.AGGRESSIVE),
                lambda: (
                    random_function(image, channel_swap, 0.1),
                    bboxes,
                ),
            ),
            (
                tf.math.equal(augmentation_config.color, ColorAugmentation.AGGRESSIVE),
                lambda: (random_function(image, grayscale, 0.1), bboxes),
            ),
            (
                tf.math.equal(augmentation_config.color, ColorAugmentation.AGGRESSIVE),
                lambda: (random_function(image, channel_drop, 0.1), bboxes),
            ),
            (
                tf.math.greater_equal(augmentation_config.color, ColorAugmentation.LIGHT),
                lambda: (tf.image.random_brightness(image, 0.2), bboxes),
            ),
            (
                tf.math.greater_equal(augmentation_config.color, ColorAugmentation.LIGHT),
                lambda: (tf.image.random_contrast(image, 0.8, 1.2), bboxes),
            ),
            (
                tf.math.greater_equal(augmentation_config.color, ColorAugmentation.MEDIUM),
                lambda: (tf.image.random_saturation(image, 0.8, 1.2), bboxes),
            ),
            (
                tf.math.greater_equal(augmentation_config.color, ColorAugmentation.MEDIUM),
                lambda: (tf.image.random_hue(image, 0.2), bboxes),
            ),
            (
                tf.math.equal(augmentation_config.crop, True),
                lambda: tf.cond(
                    tf.greater(tf.shape(bboxes)[0], 0),
                    lambda: clip_random_with_bboxes(image, bboxes),
                    lambda: (
                        clip_random(
                            image,
                            min_shape=(
                                tf.cast(tf.cast(tf.shape(image)[0], dtype=tf.float32) * 0.9, dtype=tf.int32),
                                tf.cast(tf.cast(tf.shape(image)[1], dtype=tf.float32) * 0.9, dtype=tf.int32),
                            ),
                        ),
                        bboxes,
                    ),
                ),
            ),
            (
                tf.math.equal(augmentation_config.distort_aspect_ratio, AspectRatioAugmentation.NORMAL),
                lambda: random_function_bboxes(
                    image,
                    bboxes,
                    random_aspect_ratio_deformation,
                    prob=prob_demanding_ops,
                    unify_dims=False,
                    max_squeeze=0.6,
                    max_stretch=1.3,
                ),
            ),
            (
                tf.math.equal(augmentation_config.distort_aspect_ratio, AspectRatioAugmentation.TOWARDS_SQUARE),
                lambda: random_function_bboxes(
                    image,
                    bboxes,
                    random_aspect_ratio_deformation,
                    prob=prob_demanding_ops,
                    unify_dims=True,
                    max_squeeze=0.6,
                    max_stretch=1.3,
                ),
            ),
            (
                tf.math.equal(augmentation_config.quality, True),
                lambda: (random_function(image, gaussian_noise, prob=0.15, stddev_max=0.05), bboxes),
            ),
            (
                tf.math.equal(augmentation_config.erasing, True),
                lambda: multiple_erase(
                    image,
                    bboxes,
                    iterations=tf.random.uniform((), 0, 7, tf.int32),
                    max_area=calculate_bboxes_max_erase_area(bboxes, max_area=0.1),
                ),
            ),
            (tf.math.equal(augmentation_config.rotate90, True), lambda: rot90(image, bboxes)),
            (
                tf.math.equal(augmentation_config.rotate45, True),
                lambda: (random_function_bboxes(image, bboxes, rot45, prob=0.5)),
            ),
            (
                tf.math.greater(augmentation_config.rotate_max, 0),
                lambda: random_function_bboxes(
                    image,
                    bboxes,
                    random_rotate,
                    prob=prob_demanding_ops,
                    min_rotate=-augmentation_config.rotate_max,
                    max_rotate=augmentation_config.rotate_max,
                ),
            ),
            (
                tf.math.equal(augmentation_config.flip_horizontal, True),
                lambda: random_function_bboxes(image, bboxes, flip_left_right, 0.5),
            ),
            (
                tf.math.equal(augmentation_config.flip_vertical, True),
                lambda: random_function_bboxes(image, bboxes, flip_up_down, 0.5),
            ),
        ]

        # We cannot simply index by i, this loop will find the given augmentation
        # and perform it if the precondition is satisfied.
        for i in range(len(functions)):
            image, bboxes = tf.cond(
                tf.math.logical_and(tf.equal(i, idx), functions[i][0]), functions[i][1], lambda: (image, bboxes)
            )

        return image, bboxes

    # TODO we had some problems if random_jpeg_quality was inside the random operations ... find out why
    image = tf.cond(
        tf.math.equal(augmentation_config.quality, True),
        lambda: tf.image.random_jpeg_quality(image, 35, 98),
        lambda: image,
    )

    # Randomize the sequence of augmentation indices.
    augmentation_count = 17
    order = tf.random.shuffle(tf.range(augmentation_count))

    # Loop over all augmentation and apply them.
    i = tf.constant(0, dtype=tf.int32)
    condition = lambda i, _image, _bboxes: tf.greater(augmentation_count, i)
    body = lambda i, image, bboxes: (i + 1, *apply(order[i], image, bboxes))
    _, image, bboxes = tf.while_loop(
        condition,
        body,
        (i, image, bboxes),
        shape_invariants=(
            i.get_shape(),
            tf.TensorShape([None, None, None]),
            tf.TensorShape([None, 4]),
        ),
    )

    # this ned to be at the end, otherwise we are not guaranteed to get the square
    # (and it could interact with the other augmentation in such way that we would have too much empty space)
    image, bboxes = tf.cond(
        tf.math.equal(augmentation_config.padding_square, True),
        lambda: random_pad_to_square(image, bboxes),
        lambda: (
            image,
            bboxes,
        ),
    )

    return image, bboxes
