import tensorflow as tf


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
