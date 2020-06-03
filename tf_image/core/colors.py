import tensorflow as tf

from tf_image.core.convert_type_decorator import convert_type
from tf_image.core.random import random_choice


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
