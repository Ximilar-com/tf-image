import tensorflow as tf

from tf_image.core.convert_type_decorator import convert_type
from tf_image.core.random import random_choice


@tf.function
@convert_type
def gaussian_noise(image, stddev_max=0.1):
    """
    Add a Gaussian noise into a given image.

    :param image: An image. (Float 0-1 or integer 0-255.)
    :param stddev_max: Standard deviation maximum for added Gaussian noise. Range: from 0.0 to 1.0.
    :return: Image with a Gaussian noise.
    """
    stddev = tf.random.uniform([], 0.0, stddev_max)
    noise = tf.random.normal(shape=tf.shape(image), mean=0, stddev=stddev)
    image = image + noise

    return image


@tf.function
def channel_swap(image):
    """
    Randomly swaps image channels.

    :param image: An image, last dimension is a channel.
    :return: Image with swapped channels.
    """
    indices = tf.range(start=0, limit=3, dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    image = tf.gather(image, shuffled_indices, axis=2)
    return image


@tf.function
def channel_drop(image):
    """
    Randomly drops one image channels.

    :param image: An image, last dimension is a channel.
    :return: Image with a dropped channel.
    """
    orig_dtype = image.dtype

    r, g, b = tf.split(image, 3, axis=2)
    zeros = tf.zeros_like(r, dtype=orig_dtype)

    indexes_r = tf.concat([zeros, g, b], axis=2)
    indexes_g = tf.concat([r, zeros, b], axis=2)
    indexes_b = tf.concat([r, g, zeros], axis=2)

    image = random_choice([indexes_r, indexes_g, indexes_b], 1)[0]
    return image


@tf.function
@convert_type
def rgb_shift(image, r_shift=0.0, g_shift=0.0, b_shift=0.0):
    """
    Randomly shift channels in a given image.

    :param image: An image, last dimension is a channel.
    :param r_shift: Maximal red shift delta. Range: from 0.0 to 1.0.
    :param g_shift: Maximal green shift delta. Range: from 0.0 to 1.0.
    :param b_shift: Maximal blue shift delta. Range: from 0.0 to 1.0.
    :return: Augmented image.
    """
    r, g, b = tf.split(image, 3, axis=2)

    r = r + tf.random.uniform([], -r_shift, r_shift)
    g = g + tf.random.uniform([], -g_shift, g_shift)
    b = b + tf.random.uniform([], -b_shift, b_shift)

    image = tf.concat([r, g, b], axis=2)
    return image


@tf.function
@convert_type
def grayscale(image):
    """
    Convert image to grayscale, but keep 3 dimensions.

    :param image: An image.
    :return: Grayscale image.
    """
    image = tf.image.rgb_to_grayscale(image)  # this will create one dimension
    image = tf.image.grayscale_to_rgb(image)  # this will create three dimension again
    return image


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


@tf.function
def random_resize_pad(images, height, width):
    methods = {
        0: lambda: tf.image.resize_with_pad(images, height, width, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
        1: lambda: tf.image.resize_with_pad(images, height, width, method=tf.image.ResizeMethod.BICUBIC),
        2: lambda: tf.image.resize_with_pad(images, height, width, method=tf.image.ResizeMethod.AREA),
        3: lambda: tf.image.resize_with_pad(images, height, width, method=tf.image.ResizeMethod.LANCZOS3),
        4: lambda: tf.image.resize_with_pad(images, height, width, method=tf.image.ResizeMethod.LANCZOS5),
        5: lambda: tf.image.resize_with_pad(images, height, width, method=tf.image.ResizeMethod.MITCHELLCUBIC),
        6: lambda: tf.image.resize_with_pad(images, height, width, method=tf.image.ResizeMethod.GAUSSIAN),
    }
    return tf.switch_case(tf.cast(tf.random.uniform([], 0, 1.0) * len(methods), tf.int32), branch_fns=methods)


@tf.function
def random_resize(images, height, width):
    methods = {
        0: lambda: tf.image.resize(images, (height, width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
        1: lambda: tf.image.resize(images, (height, width), method=tf.image.ResizeMethod.BICUBIC),
        2: lambda: tf.image.resize(images, (height, width), method=tf.image.ResizeMethod.AREA),
        3: lambda: tf.image.resize(images, (height, width), method=tf.image.ResizeMethod.LANCZOS3),
        4: lambda: tf.image.resize(images, (height, width), method=tf.image.ResizeMethod.LANCZOS5),
        5: lambda: tf.image.resize(images, (height, width), method=tf.image.ResizeMethod.MITCHELLCUBIC),
        6: lambda: tf.image.resize(images, (height, width), method=tf.image.ResizeMethod.GAUSSIAN),
    }
    return tf.switch_case(tf.cast(tf.random.uniform([], 0, 1.0) * len(methods), tf.int32), branch_fns=methods)
