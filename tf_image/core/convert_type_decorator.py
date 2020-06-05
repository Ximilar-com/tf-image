import tensorflow as tf
from functools import wraps


def convert_type(function):
    """
    Often, we need the input image and bounding boxes to have a specific dtype and format.
    This decorator converts image (and bounding boxes) and provides it to decorated function.
    After teh function is done, reverse conversion is done to return the same format for a user of the function.

    Image formats are the standard one accepted by TensorFlow. For Bounding boxes, we use:
    - integer types for absolute coordinates or
    - float types for relative coordinates.

    Be careful, this decorator expects specific format of function parameters:
    - without bounding boxes: image, kwargs
    - with bouding boxes: image, bounding boxes, other args, kwargs

    :param function: function to be decorated, see the requirements in the description of the decorator!
    :return: decorated function
    """

    @wraps(function)
    def wrap(image, *args, **kwargs):
        image_type = image.dtype
        image = tf.image.convert_image_dtype(image, tf.float32)

        if len(args) >= 1:
            bboxes = args[0]
            bboxes_type = bboxes.dtype
            bboxes_absolute = bboxes_type.is_integer

            bboxes = tf.cast(bboxes, tf.float32)
            bboxes = _bboxes_to_relative(image, bboxes) if bboxes_absolute else bboxes
            bboxes = tf.clip_by_value(bboxes, 0.0, 1.0)

            image, bboxes = function(image, bboxes, *args[1:], **kwargs)
            image = tf.clip_by_value(image, 0.0, 1.0)
            image = tf.image.convert_image_dtype(image, image_type, saturate=True)

            bboxes = tf.clip_by_value(bboxes, 0.0, 1.0)
            bboxes = _bboxes_to_absolute(image, bboxes) if bboxes_absolute else bboxes
            bboxes = tf.cast(bboxes, bboxes_type)

            return image, bboxes

        image = function(image, **kwargs)
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = tf.image.convert_image_dtype(image, image_type, saturate=True)
        return image

    return wrap


@tf.function
def _bboxes_to_relative(image, bboxes):
    image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
    bboxes_update = tf.cast(tf.stack([image_height, image_width, image_height, image_width]), dtype=tf.float32)
    return bboxes / bboxes_update


@tf.function
def _bboxes_to_absolute(image, bboxes):
    image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
    bboxes_update = tf.cast(tf.stack([image_height, image_width, image_height, image_width]), dtype=tf.float32)
    return bboxes * bboxes_update
