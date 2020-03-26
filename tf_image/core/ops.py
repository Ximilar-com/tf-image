import tensorflow as tf

from tf_image.core.random import random_choice, random_function


@tf.function
def gaussian_noise(image, stddev_max=0.1):
    stddev = tf.random.uniform([], 0.0, stddev_max)
    noise = tf.random.normal(shape=tf.shape(image), mean=0, stddev=stddev)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)
    return tf.cast(image, tf.float32)


@tf.function
def channel_swap(image):
    indices = tf.range(start=0, limit=3, dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    zeros = tf.zeros(tf.shape(image))
    image = tf.gather(image, shuffled_indices, axis=2)
    return image


@tf.function
def channel_drop(image):
    height, width, channels = image.shape

    image = tf.cast(image, dtype=tf.float32)
    r, g, b = tf.split(image, 3, axis=2)
    zeros = tf.zeros_like(r, dtype=tf.float32)

    indexes_r = tf.concat([zeros, g, b], axis=2)
    indexes_g = tf.concat([r, zeros, b], axis=2)
    indexes_b = tf.concat([r, g, zeros], axis=2)

    image = random_choice([indexes_r, indexes_g, indexes_b], 1)[0]
    return image


@tf.function
def rgb_shift(image, r_shift=0.0, g_shift=0.0, b_shift=0.0):
    image = tf.cast(image, dtype=tf.float32)
    r, g, b = tf.split(image, 3, axis=2)

    r = r + tf.random.uniform([], -r_shift, r_shift)
    g = g + tf.random.uniform([], -g_shift, g_shift)
    b = b + tf.random.uniform([], -b_shift, b_shift)

    image = tf.concat([r, g, b], axis=2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def grayscale(image):
    image = tf.image.rgb_to_grayscale(image)  # this will create one dimension
    image = tf.image.grayscale_to_rgb(image)  # this will create three dimension again
    return image
