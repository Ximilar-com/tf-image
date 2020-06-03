import tensorflow as tf

from tf_image.core.clip import clip_random
from tf_image.core.convert_type_decorator import convert_type


@tf.function
@convert_type
def clip_random_with_bboxes(image, bboxes, min_shape=(1, 1)):
    """
    Randomly clips an image in such way, that all bounding boxes are still fully present in the resulting image.
    Update bounding boxes to match the new image.

    :param image: 3-D Tensor of shape (height, width, channels).
    :param bboxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [ymin, xmin, ymin, xmax]
    :return: (clipped image, updated bounding boxes)
    """
    with tf.name_scope("clip_random_with_bboxes"):
        return tf.cond(
            tf.equal(tf.reduce_sum(bboxes), 0),
            lambda: (clip_random(image, min_shape), bboxes),
            lambda: _clip_random_with_bboxes(image, bboxes),
        )


@tf.function
def _clip_random_with_bboxes(image, bboxes):
    image_height = tf.cast(tf.shape(image)[0], dtype=tf.float32)
    image_width = tf.cast(tf.shape(image)[1], dtype=tf.float32)

    # calculate coordinates
    new_miny = tf.random.uniform([], 0.0, tf.math.reduce_min(bboxes[:, 0]))
    new_minx = tf.random.uniform([], 0.0, tf.math.reduce_min(bboxes[:, 1]))
    new_maxy = tf.random.uniform([], tf.math.reduce_max(bboxes[:, 2]), 1.0)
    new_maxx = tf.random.uniform([], tf.math.reduce_max(bboxes[:, 3]), 1.0)
    new_height, new_width = new_maxy - new_miny, new_maxx - new_minx

    # prepare parameters
    args_clip_bboxes = [new_miny, new_minx, new_height, new_width]
    args_clip_image = [
        tf.cast(new_miny * image_height, dtype=tf.int32),
        tf.cast(new_minx * image_width, dtype=tf.int32),
        tf.cast(new_height * image_height, dtype=tf.int32),
        tf.cast(new_width * image_width, dtype=tf.int32),
    ]

    # update
    image, bboxes = tf.cond(
        tf.math.logical_or(tf.math.greater(new_height, 0), tf.math.greater(new_width, 0)),
        lambda: (tf.image.crop_to_bounding_box(image, *args_clip_image), clip_bboxes(bboxes, *args_clip_bboxes)),
        lambda: (image, bboxes),
    )

    return image, bboxes


@tf.function
def clip_bboxes(bboxes_relative, new_miny, new_minx, new_height, new_width):
    """
    Calculates new coordinates for given bounding boxes given the cut area of an image.

    :param bboxes_relative: 2-D Tensor (box_number, 4) containing bounding boxes in format [ymin, xmin, ymin, xmax].
    :param new_miny: Relative clipping coordinate.
    :param new_minx: Relative clipping coordinate.
    :param new_height: Relative clipping coordinate.
    :param new_width: Relative clipping coordinate.
    :return: clipped bounding boxes
    """
    # move the coordinates according to new min value
    bboxes_move_min = tf.stack([new_miny, new_minx, new_miny, new_minx])
    bboxes = bboxes_relative - bboxes_move_min

    # if we use relative coordinates, we have to scale the coordinates to be between 0 and 1 again
    bboxes_scale = [new_height, new_width, new_height, new_width]
    bboxes = bboxes / bboxes_scale

    return bboxes
