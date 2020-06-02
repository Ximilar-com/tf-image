import tensorflow as tf

from tf_image.core.convert_type_decorator import convert_type


@tf.function
@convert_type
def flip_left_right(image, bboxes):
    """
    Flip an image and bounding boxes horizontally (left to right).

    :param image: 3-D Tensor of shape [height, width, channels]
    :param bboxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [ymin, xmin, ymin, xmax]
    :return: image, bounding boxes
    """
    with tf.name_scope("flip_left_right"):
        bboxes = bboxes * tf.constant([1, -1, 1, -1], dtype=tf.float32) + tf.stack([0.0, 1.0, 0.0, 1.0])
        bboxes = tf.stack([bboxes[:, 0], bboxes[:, 3], bboxes[:, 2], bboxes[:, 1]], axis=1)

        image = tf.image.flip_left_right(image)

    return image, bboxes


@tf.function
@convert_type
def flip_up_down(image, bboxes):
    """
    Flip an image and bounding boxes vertically (upside down).

    :param image: 3-D Tensor of shape [height, width, channels]
    :param bboxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [ymin, xmin, ymin, xmax]
    :return: image, bounding boxes
    """
    with tf.name_scope("flip_up_down"):
        bboxes = bboxes * tf.constant([-1, 1, -1, 1], dtype=tf.float32) + tf.stack([1.0, 0.0, 1.0, 0.0])
        bboxes = tf.stack([bboxes[:, 2], bboxes[:, 1], bboxes[:, 0], bboxes[:, 3]], axis=1)

        image = tf.image.flip_up_down(image)

    return image, bboxes
