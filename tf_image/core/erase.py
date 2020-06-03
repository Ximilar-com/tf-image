import tensorflow as tf

from tf_image.core.convert_type_decorator import convert_type


@tf.function
@convert_type
def random_erasing(image, max_area=0.1, erased_value=0):
    """
    Randomly removes rectangular part of given image.

    :param image: An image. (Float 0-1 or integer 0-255.)
    :param max_area: Maximum part of the image to be erased. (Range: 0.0 to 1.0)
    :param erased_value: The value which will be used for the empty area.
    :return: Augmented image.
    """
    image_height, image_width = tf.shape(image)[-3], tf.shape(image)[-2]
    max_area = tf.cast(max_area * tf.cast(image_height * image_width, tf.float32), tf.int32)

    return tf.cond(tf.greater_equal(max_area, 1), lambda: _random_erasing(image, max_area, erased_value), lambda: image)


@tf.function
def _random_erasing(image, max_area, erased_value):
    image_height, image_width = tf.shape(image)[-3], tf.shape(image)[-2]

    # Get center of the rectangle to be removed.
    y = tf.random.uniform([], 1, image_height - 2, dtype=tf.int32)
    x = tf.random.uniform([], 1, image_width - 2, dtype=tf.int32)

    # Functions fo calculating the size of the erased space.
    def random(max_val):
        return tf.cond(tf.greater(max_val, 1), lambda: tf.random.uniform([], 1, max_val, dtype=tf.int32), lambda: 1)

    def get_size(center1, center2, max1, max2):
        size1 = random(tf.math.reduce_min([center1, max1 - center1, max_area]))
        size2 = random(tf.math.reduce_min([center2, max2 - center2, max_area // size1]))
        return size1, size2

    def swap(size1, size2):
        return size2, size1

    # If we use only one of those, we would get a lot vertical / horizontal rectangles.
    # Changing the first generated size, we fix the distribution.
    height, width = tf.cond(
        tf.math.greater(tf.random.uniform([], 0, 1), 0.5),
        lambda: get_size(y, x, image_height, image_width),
        lambda: swap(*get_size(x, y, image_width, image_height)),
    )

    # Crate mask for generated rectangle.
    mask = tf.ones((height, width), dtype=image.dtype)
    top, left = y - height // 2, x - width // 2
    mask = tf.pad(mask, [[top, image_height - top - height], [left, image_width - left - width]])
    mask = tf.image.grayscale_to_rgb(tf.expand_dims(mask, -1))

    # Now, we can erase the rectangle from the image.
    image = image * (1 - mask) + mask * erased_value
    return image
