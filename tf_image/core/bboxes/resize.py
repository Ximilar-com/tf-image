import tensorflow as tf

from tf_image.core.convert_type_decorator import convert_type
from tf_image.core.resize import random_resize_pad, random_resize


@convert_type
@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.bool),
        tf.TensorSpec(shape=(), dtype=tf.bool),
    ]
)
def resize(image, bboxes, height, width, keep_aspect_ratio=True, random_method=False):
    """
    Resize given image and bounding boxes.

    :param image: 3-D Tensor of shape [height, width, channels].
    :param bboxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [ymin, xmin, ymin, xmax]
    :param height: Height of the resized image.
    :param width: Width of the resized image.
    :param keep_aspect_ratio: True if we should add padding instead to fully spreading out the image to given size
    :param random_method: Whether we should use random resize method (tf.image.ResizeMethod) or the default one.
    :return: (resized image, resized bounding boxes)
    """
    with tf.name_scope("resize"):

        def _keep_aspect_ratio(img, boxes, h, w):
            image_shape = tf.cast(tf.shape(img), tf.float32)
            image_height, image_width = image_shape[0], image_shape[1]

            img = tf.cond(
                random_method,
                lambda: random_resize_pad(img, height, w),
                lambda: tf.image.resize_with_pad(img, height, w),
            )

            h, w = tf.cast(h, dtype=tf.float32), tf.cast(w, dtype=tf.float32)
            resize_coef = tf.math.minimum(h / image_height, w / image_width)
            resized_height, resized_width = image_height * resize_coef, image_width * resize_coef
            pad_y, pad_x = (h - resized_height) / 2, (w - resized_width) / 2
            boxes = boxes * tf.stack([resized_height, resized_width, resized_height, resized_width]) + tf.stack(
                [pad_y, pad_x, pad_y, pad_x,]
            )

            boxes /= tf.stack([h, w, h, w])

            return img, boxes

        def _dont_keep_aspect_ration(img, boxes, h, w):
            img = tf.cond(random_method, lambda: random_resize(img, h, w), lambda: tf.image.resize(img, (h, w)),)

            return img, boxes

        image, bboxes = tf.cond(
            keep_aspect_ratio,
            lambda: _keep_aspect_ratio(image, bboxes, height, width),
            lambda: _dont_keep_aspect_ration(image, bboxes, height, width),
        )
    return image, bboxes


@tf.function
@convert_type
def random_aspect_ratio_deformation(image, bboxes, max_squeeze=0.7, max_stretch=1.3, unify_dims=False):
    """
    Randomly pick width or height dimension and squeeze or stretch given image in that dimension.

    Often, we train on a square images. To fill bigger part of this square, we can set parameter unify_dims to True.
    This will allow us to stretch the short side / squeeze the long size by a bigger ration that max_squeeze/max_stretch
    and get more squared image.

    :param image: 3-D Tensor of shape [height, width, channels].
    :param bboxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [ymin, xmin, ymin, xmax]
    :param max_squeeze: Maximum relative coefficient for squeezing an image size. (0.0 to 1.0)
    :param max_stretch: Maximum relative coefficient for stretching an image size. (0.0 to 1.0)
    :param unify_dims: overwrite max_squeeze of long side and max_stretch of short size to be able to fill a square
    :return: (augmented image, updated bounding boxes)
    """
    with tf.name_scope("random_aspect_ratio_deformation"):
        image_shape = tf.cast(tf.shape(image), dtype=tf.float32)
        height, width = image_shape[0], image_shape[1]

        # Do we do the squeeze/stretch the y or x side?
        side = tf.random.uniform([], 0, 2, dtype=tf.int32)

        # update max squeeze / stretch of unify_dims is set to true
        # (if parameters can get bigger in order to fit the square better)
        max_squeeze_h = tf.math.maximum(max_squeeze, height / width) if unify_dims else max_stretch
        max_squeeze_w = tf.math.maximum(max_squeeze, width / height) if unify_dims else max_stretch
        max_stretch_h = tf.math.maximum(max_stretch, width / height) if unify_dims else max_stretch
        max_stretch_w = tf.math.maximum(max_stretch, height / width) if unify_dims else max_stretch

        # new size
        height = height * tf.cond(side == 0, lambda: tf.random.uniform([], max_squeeze_h, max_stretch_h), lambda: 1.0)
        width = width * tf.cond(side != 0, lambda: tf.random.uniform([], max_squeeze_w, max_stretch_w), lambda: 1.0)
        height, width = tf.cast(height, dtype=tf.int32), tf.cast(width, dtype=tf.int32)

        image, bboxes = resize(image, bboxes, height, width, keep_aspect_ratio=False)

    return image, bboxes


@tf.function
@convert_type
def random_pad_to_square(
    image, bboxes, max_extend=0.1,
):
    """
    Creates a square image from a given input. The final size is given by the longer input image size + random padding
    limited by max_extend parameter. The position of the original image inside this square is random as well.

    :param image: 3-D Tensor of shape [height, width, channels].
    :param bboxes: 2-D Tensor (box_number, 4) containing bounding boxes in format [ymin, xmin, ymin, xmax]
    :param max_extend:  maximal free space that could be added to the bigger side of the image
    :return: (padding image, updated bounding boxes)
    """
    with tf.name_scope("random_pad_to_square"):
        height, width = tf.shape(image)[0], tf.shape(image)[1]

        # how much empty space we add to the longer side
        max_extend = tf.math.maximum(height, width) // tf.cast(100 * max_extend, dtype=tf.int32)
        extend = tf.cond(
            tf.math.greater_equal(max_extend, 1),
            lambda: tf.random.uniform([], 0, max_extend, dtype=tf.int32),
            lambda: 0,
        )

        # find out, how much we can extend the longer side and shift the shorted side at most
        max_padding_top = tf.math.maximum(extend, width - height)
        max_padding_left = tf.math.maximum(extend, height - width)

        # now, take a random padding values
        padding_top = tf.cond(
            tf.math.greater(max_padding_top, 1),
            lambda: tf.random.uniform([], 0, max_padding_top, dtype=tf.int32),
            lambda: 0,
        )
        padding_left = tf.cond(
            tf.math.greater(max_padding_left, 1),
            lambda: tf.random.uniform([], 0, max_padding_left, dtype=tf.int32),
            lambda: 0,
        )

        # this will be the final width and height of the image
        size = tf.math.maximum(height + padding_top, width + padding_left)

        # pad image on all sides to get a square image
        image = tf.pad(
            image, [[padding_top, size - padding_top - height], [padding_left, size - padding_left - width], [0, 0]]
        )

        # we need to cast all dimensions to float co continue with bounding box calculations
        height, width = tf.cast(height, dtype=tf.float32), tf.cast(width, dtype=tf.float32)
        padding_top, padding_left = tf.cast(padding_top, dtype=tf.float32), tf.cast(padding_left, dtype=tf.float32)
        size = tf.cast(size, dtype=tf.float32)

        # update positions of bounding boxes
        padding = tf.stack([padding_top, padding_left, padding_top, padding_left,])
        bboxes = bboxes * tf.stack([height, width, height, width]) + padding

        bboxes /= tf.stack([size, size, size, size])

    return image, bboxes
