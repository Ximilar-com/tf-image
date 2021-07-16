import tensorflow as tf
import tensorflow_addons as tfa

from math import pi

from tf_image.core.bboxes.clip import clip_bboxes
from tf_image.core.convert_type_decorator import convert_type


@tf.function
@convert_type
def random_rotate(image, bboxes, min_rotate=-20, max_rotate=20):
    """
    Randomly Rotates image and bonding boxes.
    The rotation degree is chosen from provided range with an uniform probability.

    We do not cut any part of the image. Zeros padding is added around to fill the empty space after rotation.

    Rotating bounding boxes has one significant drawback. The result (except few special cases)
    is not a proper bounding box! The ages are not vertical and horizontal. We need to fix that.
    We took the same approach as with the whole image. To be sure no part of the object is left out,
    we make a bounding box around the rotated bounding box. Unfortunately this approach means that
    we also increase its size.

     ###########################
     ##            ###        ##
     ##          ##    ####   ##
     ##        ##          ## ##
     ##       ##         ##   ##
     ##      ##         ##    ##
     ##     ##         ##     ##
     ##   ##          ##      ##
     ##  ##         ##        ##
     ## ##         ##         ##
     ##   ####    ##          ##
     ##        ###            ##
     ###########################

    :param image: 3-D Tensor of shape (height, width, channels).
    :param bboxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [ymin, xmin, ymin, xmax].
    :param min_rotate: Maximal scalar angle in degrees that can be used for counterclockwise rotation.
    :param max_rotate: Minimal scalar angle in degrees that can be used for counterclockwise rotation.
    :return: (rotated image, rotated bounding boxes)
    """
    if min_rotate == 0 and max_rotate == 0:
        return image, bboxes

    if min_rotate >= max_rotate:
        raise ValueError(f"Minimum has to be greater than maximum! {min_rotate} {max_rotate}")

    with tf.name_scope("random_rotate"):
        rotate_mean = (min_rotate + max_rotate) / 2
        rotate_stdev = (max_rotate - min_rotate) / 4
        rotate = tf.random.truncated_normal([], mean=rotate_mean, stddev=rotate_stdev)
        image, bboxes = _rotate(image, bboxes, rotate)

    return image, bboxes


@tf.function
def _rotate(image, bboxes, angle):
    image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
    rotate = angle * pi / 180 + tf.keras.backend.epsilon()

    # find the new width and height bounds
    abs_cos = tf.math.abs(tf.math.cos(rotate))
    abs_sin = tf.math.abs(tf.math.sin(rotate))
    bound_h = tf.cast(
        tf.cast(image_height, dtype=tf.float32) * abs_cos + tf.cast(image_width, dtype=tf.float32) * abs_sin,
        dtype=tf.int32,
    )
    bound_w = tf.cast(
        tf.cast(image_height, dtype=tf.float32) * abs_sin + tf.cast(image_width, dtype=tf.float32) * abs_cos,
        dtype=tf.int32,
    )

    # if the new bounds are bigger than the old ones on some side, add some padding
    pad_bound_h = tf.math.maximum(image_height, bound_h)
    pad_bound_w = tf.math.maximum(image_width, bound_w)
    pad_y = (pad_bound_h - image_height) // 2
    pad_x = (pad_bound_w - image_width) // 2

    image = tf.image.pad_to_bounding_box(image, pad_y, pad_x, pad_bound_h, pad_bound_w)

    bboxes_resize = [image_height, image_width, image_height, image_width]
    bboxes_pad = [pad_y, pad_x, pad_y, pad_x]
    bboxes = tf.cast(bboxes * bboxes_resize + bboxes_pad, dtype=tf.float32)
    bboxes_points = tf.map_fn(_unpack_bbox, bboxes)
    bboxes_points = _rotate_points(bboxes_points, -rotate, image)
    bboxes = tf.map_fn(
        _find_bbox,
        bboxes_points,
    )

    image = tfa.image.rotate(image, rotate)
    image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]

    bboxes_resize = tf.cast(tf.stack([image_height, image_width, image_height, image_width]), dtype=tf.float32)
    bboxes = bboxes / bboxes_resize

    # if the new bounds are smaller than the old ones on some side, remove the empty space
    clip_y, clip_x = image_height - bound_h, image_width - bound_w
    clip_args = [clip_y // 2, clip_x // 2, image_height - clip_y, image_width - clip_x]
    clip_args_rel = [
        tf.cast(clip_args[0] / image_height, dtype=tf.float32),
        tf.cast(clip_args[1] / image_width, dtype=tf.float32),
        tf.cast(clip_args[2] / image_height, dtype=tf.float32),
        tf.cast(clip_args[3] / image_width, dtype=tf.float32),
    ]
    image, bboxes = tf.cond(
        tf.math.logical_or(tf.math.greater(clip_y, 0), tf.math.greater(clip_x, 0)),
        lambda: (tf.image.crop_to_bounding_box(image, *clip_args), clip_bboxes(bboxes, *clip_args_rel)),
        lambda: (image, bboxes),
    )

    return image, bboxes


@tf.function
def _unpack_bbox(bbox):
    """
    Translate bounding box into corner coordinates.

    :param bbox: Bounding box of a shape [ymin, xmin, ymax, xmax].
    :return: List of corner coordinates.
    """
    ymin, xmin, ymax, xmax = bbox[0], bbox[1], bbox[2], bbox[3]
    return tf.stack([[ymin, xmin], [ymin, xmax], [ymax, xmin], [ymax, xmax]])


@tf.function
def _find_bbox(points):
    """
    Return smallest bounding box containing all given points.

    :param points: List of 2D points [y, x],
    :return: Bounding box of a shape [ymin, xmin, ymax, xmax].
    """
    return tf.stack(
        [
            tf.math.reduce_min(points[:, 0]),
            tf.math.reduce_min(points[:, 1]),
            tf.math.reduce_max(points[:, 0]),
            tf.math.reduce_max(points[:, 1]),
        ]
    )


@tf.function
def _rotate_points(points, angle, image):
    """
    Rotate all points in a given list around a center of given image.

    :param points: List of 2D points [y, x].
    :param angle: Angle in radians.
    :param image: A reference image.
    :return: List of rotated points.
    """
    image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
    center = tf.cast(tf.stack([image_height / 2, image_width / 2]), dtype=tf.float32)
    rotation_matrix = tf.stack([tf.math.cos(angle), -tf.math.sin(angle), tf.math.sin(angle), tf.math.cos(angle)])
    rotation_matrix = tf.reshape(rotation_matrix, (2, 2))
    points = tf.matmul(points - center, rotation_matrix) + center
    return points


@tf.function
@convert_type
def rot90(image, bboxes, k=(0, 1, 2, 3)):
    """
    Rotate image and bounding boxes counter-clockwise by random multiple of 90 degrees.

    :param image: 3-D Tensor of shape [height, width, channels]
    :param bboxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [ymin, xmin, ymin, xmax]
    :param k: array with multiples of 90 to choose from
    :return: (rotated image, rotated bounding boxes)
    """
    with tf.name_scope("rot90"):
        selected_k = tf.math.floormod(tf.random.shuffle(k)[0], 4)

        image = tf.image.rot90(image, k=selected_k)

        rotate_bboxes = [
            lambda: bboxes,
            lambda: tf.stack(
                [tf.math.subtract(1.0, bboxes[:, 3]), bboxes[:, 0], tf.math.subtract(1.0, bboxes[:, 1]), bboxes[:, 2]],
                axis=1,
            ),
            lambda: tf.math.subtract(
                1.0,
                tf.stack(
                    [
                        bboxes[:, 2],
                        bboxes[:, 3],
                        bboxes[:, 0],
                        bboxes[:, 1],
                    ],
                    axis=1,
                ),
            ),
            lambda: tf.stack(
                [bboxes[:, 1], tf.math.subtract(1.0, bboxes[:, 2]), bboxes[:, 3], tf.math.subtract(1.0, bboxes[:, 0])],
                axis=1,
            ),
        ]

        bboxes = tf.cond(
            tf.greater(tf.shape(bboxes)[0], 0),
            lambda: tf.switch_case(selected_k, rotate_bboxes),
            lambda: bboxes,
        )

    return image, bboxes


@tf.function
@convert_type
def rot45(image, bboxes):
    """
    Rotate image and bounding boxes counter-clockwise by 45 degrees.

    :param image: 3-D Tensor of shape [height, width, channels]
    :param bboxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [ymin, xmin, ymin, xmax]
    :return: (rotated image, rotated bounding boxes)
    """
    with tf.name_scope("rot45"):
        return _rotate(image, bboxes, 45)
