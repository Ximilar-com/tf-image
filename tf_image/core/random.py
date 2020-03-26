import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops


@tf.function
def random_choice(x, size, axis=0):
    dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
    indices = tf.range(0, dim_x, dtype=tf.int64)
    sample_index = tf.random.shuffle(indices)[:size]
    sample = tf.gather(x, sample_index, axis=axis)
    return sample


@tf.function
def random_function(image, function, prob, seed, **kwargs):
    with ops.name_scope(None, "random_" + function.__name__, [image]) as scope:
        uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
        image = ops.convert_to_tensor(image, name="image")
        mirror_cond = math_ops.less(uniform_random, prob)
        result = control_flow_ops.cond(mirror_cond, lambda: function(image, **kwargs), lambda: image, name=scope)
    return result
