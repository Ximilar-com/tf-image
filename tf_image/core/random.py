import tensorflow as tf


@tf.function
def random_choice(x, size, axis=0):
    dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
    indices = tf.range(0, dim_x, dtype=tf.int64)
    sample_index = tf.random.shuffle(indices)[:size]
    sample = tf.gather(x, sample_index, axis=axis)
    return sample


def random_function(image, function, prob, seed=None, **kwargs):
    with tf.name_scope("random_" + function.__name__):
        uniform_random = tf.random.uniform([], 0, 1.0, seed=seed)
        mirror_cond = tf.math.less(uniform_random, prob)
        result = tf.cond(mirror_cond, lambda: function(image, **kwargs), lambda: image)
    return result


def random_function_bboxes(image, bboxes, function, prob, seed=None, **kwargs):
    with tf.name_scope("random_" + function.__name__):
        uniform_random = tf.random.uniform([], 0, 1.0, seed=seed)
        mirror_cond = tf.math.less(uniform_random, prob)
        result = tf.cond(mirror_cond, lambda: function(image, bboxes, **kwargs), lambda: (image, bboxes))
    return result