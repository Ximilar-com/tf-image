import tensorflow as tf
import unittest

from tf_image.core.bboxes.clip import clip_bboxes, clip_random_with_bboxes
from tf_image.core.bboxes.erase import multiple_erase
from tf_image.core.bboxes.flip import (
    flip_left_right,
    flip_up_down,
)
from tf_image.core.bboxes.resize import resize, random_pad_to_square, random_aspect_ratio_deformation
from tf_image.core.bboxes.rotate import _find_bbox, _unpack_bbox


class BboxTest(unittest.TestCase):
    def check_types(self, function):
        """
        We want our functions to work with different types to provided user the freedom to choose.
        We test the two standard ones - tf.int8, tf.float32. If this works, other meaningful types are OK as well.

        We provide the image and bounding boxes. Function does not need to use both or return meaningful value for both.
        But it has to return at least None.

        :param function: (image, bounding boxes, relative coordinates) -> (image, bounding boxes)
        """
        dtypes = [tf.int8, tf.float32]
        for dtype in dtypes:
            image = tf.ones([24, 8, 3], dtype=dtype)
            bboxes = tf.constant([[1, 1, 15, 6]], dtype=dtype)

            if dtype.is_floating:
                bboxes /= tf.cast([image.shape[0], image.shape[1], image.shape[0], image.shape[1]], dtype)

            image, bboxes = function(image, bboxes)

            if image is not None:
                self.assertEqual(image.dtype, dtype, msg="Image type does not fit.")

            if bboxes is not None:
                self.assertEqual(bboxes.dtype, dtype, msg="Bounding box type does not fit.")


class TestClip(BboxTest):
    def test_clip_types(self):
        self.check_types(lambda image, bboxes: clip_random_with_bboxes(image, bboxes))

    def test_clip_relative(self):
        # calculation done for  image = tf.ones([24, 8, 3])
        bboxes = tf.constant([[0.1, 0.25, 0.5, 0.75]])

        clipped = clip_bboxes(bboxes, tf.constant(0.1), tf.constant(0.125), tf.constant(0.5), tf.constant(0.75))
        tf.debugging.assert_equal(tf.constant([[0.0, 1.0 / 6.0, 0.8, 5.0 / 6.0]]), clipped)


class TestErase(BboxTest):
    def test_flip_left_right_types(self):
        self.check_types(lambda image, bboxes: multiple_erase(image, bboxes))


class TestFlip(BboxTest):
    def test_flip_left_right_types(self):
        self.check_types(lambda image, bboxes: flip_left_right(image, bboxes))

    def test_flip_left_right_absolute(self):
        image = tf.ones([24, 8, 3])
        bboxes = tf.constant([[1, 1, 15, 6]],)

        _, flipped = flip_left_right(image, bboxes)
        tf.debugging.assert_equal(tf.constant([[1, 2, 15, 7]]), flipped)

    def test_flip_left_right_relative(self):
        image = tf.ones([24, 8, 3])
        bboxes = tf.constant([[0.1, 0.2, 0.4, 0.7]])

        _, flipped = flip_left_right(image, bboxes)
        tf.debugging.assert_equal(tf.constant([[0.1, 0.3, 0.4, 0.8]]), flipped)

    def test_flip_up_down_types(self):
        self.check_types(lambda image, bboxes: flip_up_down(image, bboxes))

    def test_flip_up_down_absolute(self):
        image = tf.ones([24, 8, 3])
        bboxes = tf.constant([[1, 1, 15, 6]])

        _, flipped = flip_up_down(image, bboxes)
        tf.debugging.assert_equal(tf.constant([[9, 1, 23, 6]]), flipped)

    def test_flip_up_down_relative(self):
        image = tf.ones([24, 8, 3])
        bboxes = tf.constant([[0.1, 0.2, 0.4, 0.7]])

        _, flipped = flip_up_down(image, bboxes)
        tf.debugging.assert_equal(tf.constant([[0.6, 0.2, 0.9, 0.7]]), flipped)


class TestResize(BboxTest):
    def setUp(self):
        self.image = tf.ones([24, 8, 3])
        self.bboxes = tf.constant([[0.25, 0.50, 0.75, 0.8]])

    def test_resize_types(self):
        self.check_types(lambda image, bboxes: resize(image, bboxes, 20, 20))
        self.check_types(lambda image, bboxes: random_pad_to_square(image, bboxes))
        self.check_types(lambda image, bboxes: random_aspect_ratio_deformation(image, bboxes))

    def test_resize_keep_aspect_ratio(self):
        image_resized, bboxes_resized = resize(self.image, self.bboxes, 6, 6, keep_aspect_ratio=True)
        tf.debugging.assert_equal(
            tf.constant([6, 6, 3]), image_resized.shape, message="Dimensions of the resized image are wrong."
        )

        tf.debugging.assert_equal(
            tf.round(tf.ones([6, 2, 3], dtype=tf.float32) * 1000),
            tf.round(image_resized[:, 2:4, :] * 1000),
            message="Ones from original image should be there.",
        )
        tf.debugging.assert_equal(tf.zeros([6, 2, 3]), image_resized[:, 4:, :], message="Padding should be here.")
        tf.debugging.assert_equal(tf.zeros([6, 2, 3]), image_resized[:, :2, :], message="Padding should be here.")

        h, w = self.image.shape[0], self.image.shape[1]
        tf.debugging.assert_equal(tf.constant([[0.25, 0.5, 0.75, 0.6]]), bboxes_resized)

    def test_resize_not_keep_aspect_ratio(self):
        image_resized, bboxes_resized = resize(self.image, self.bboxes, 6, 6, keep_aspect_ratio=False)
        tf.debugging.assert_equal(
            tf.constant([6, 6, 3]), image_resized.shape, message="Dimensions of the resized image are wrong."
        )

        mult = 10
        tf.debugging.assert_equal(
            tf.ones([6, 6, 3]) * mult,
            tf.math.round(image_resized * mult),
            message="New image should contain only ones.",
        )

        h, w = self.image.shape[0], self.image.shape[1]
        tf.debugging.assert_equal(self.bboxes, bboxes_resized)


class TestRotate(unittest.TestCase):
    def test_find_bbox(self):
        coordinates = tf.constant(
            [[10.0, 2.0], [15.0, 6.0], [22.0, 10.0], [1.0, 30.0], [5.0, 2.0], [8.0, 9.0], [10.0, 15.0], [22.0, 3.0]]
        )
        bbox = _find_bbox(coordinates)
        tf.debugging.assert_equal(tf.constant([1.0, 2.0, 22.0, 30.0]), bbox)

    def test_pack_unpack(self):
        bbox = tf.constant([1, 2, 3, 4])
        bbox2 = _find_bbox(_unpack_bbox(bbox))
        tf.debugging.assert_equal(bbox, bbox2)


if __name__ == "__main__":
    unittest.main()
