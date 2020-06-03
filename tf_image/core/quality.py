import tensorflow as tf

from tf_image.core.convert_type_decorator import convert_type


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
