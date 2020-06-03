import tensorflow as tf

from tf_image.core.convert_type_decorator import convert_type
from tf_image.core.erase import random_erasing


@convert_type
def multiple_erase(image, bboxes, iterations=10, max_area=1.0):
    """
    Repeatedly erase rectangular areas from given image.

    :param image: 3-D Tensor of shape [height, width, channels].
    :param bboxes: Bounding box representing the cut which will give us the clipped image.
    :param iterations: How many random rectangles we are going to erase.
    :param max_area: Maximum part of the image to be erased in one iteration. (Range: 0.0 to 1.0)
    :return: (augmented image, unchanged bboxes)
    """
    with tf.name_scope("multiple_erase"):
        max_area = tf.clip_by_value(tf.cast(max_area, dtype=tf.float32), 0.0, 1.0)

        i = tf.constant(0)
        condition = lambda i, _image: i < iterations
        body = lambda i, image: (i + 1, random_erasing(image, max_area=max_area))
        _, image = tf.while_loop(condition, body, (i, image))

    return image, bboxes


def calculate_bboxes_max_erase_area(bboxes, max_area=1.0, erase_smallest=0.5):
    """
    Calculates the the biggest area (width * height) that can be erased on an image with given bounding boxes.

    Result = smallest value from (smallest bounding boxes size *  erase_smallest) or max_area

    :param bboxes: Bounding box representing the cut which will give us the clipped image.
    :param max_area: Maximum part of the image to be erased in one iteration. (0.0 to 1.0)
    :param erase_smallest: Multiple of the smallest bounding that we could erase. (0.0 none, 1.0 full or more.)
    :return: relative max_area (0.0 - 1.0)
    """
    max_area = tf.clip_by_value(tf.cast(max_area, dtype=tf.float32), 0.0, 1.0)

    sizes = bboxes[:, 2:] - bboxes[:, :2]
    areas = tf.math.reduce_prod(sizes, axis=1)
    smallest_bbox_area = tf.math.reduce_min(areas)  # inf if there are no bonding boxes
    return tf.minimum(smallest_bbox_area * (1 - erase_smallest), max_area)
