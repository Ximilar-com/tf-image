import tensorflow as tf


@tf.function
def clip_random(image, min_shape):
    """
    Randomly cuts out part of the image. Useful for images with no bounding boxes. It provides additional parameter
    for minimum size which is not needed when we have bounding boxes.

    If the height or width of an image is smaller than min_shape, we keep the given dimension.

    :param image: 3-D Tensor of shape (height, width, channels).
    :param min_shape: smallest image cut size, (height, width).
    :return: clipped image
    """
    img_height, img_width = tf.shape(image)[0], tf.shape(image)[1]
    min_height, min_width = min_shape[0], min_shape[1]

    height = tf.cond(
        tf.math.greater(img_height, min_height),
        lambda: tf.random.uniform([], min_height, img_height, dtype=tf.int32),
        lambda: img_height,
    )
    width = tf.cond(
        tf.math.greater(img_width, min_width),
        lambda: tf.random.uniform([], min_width, img_width, dtype=tf.int32),
        lambda: img_width,
    )

    image = tf.image.random_crop(image, size=(height, width, tf.shape(image)[-1]))
    return image


