import tensorflow as tf

from tf_image.application.augmentation_config import AugmentationConfig
from tf_image.application.tools import random_augmentations

#
# Loads an images from images/ximilar-similar.jpg, create some bounding boxes and augment
# three times without bounding boxes and three times with them. Results are saved to images/results folder.
#

# Loads the basic setup, feel free to experiment!
config = AugmentationConfig()

# Loads the image and creates bounding boxes for three completely visible apples.
image_encoded = tf.io.read_file("images/ximilar-similar.jpg")
image = tf.image.decode_jpeg(image_encoded)

bboxes = tf.constant([[262.0, 135.0, 504.0, 371.0], [285.0, 446.0, 494.0, 644.0], [272.0, 688.0, 493.0, 895.0]])
bboxes /= tf.cast(
    tf.stack([tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[0], tf.shape(image)[1],]), tf.float32
)
bboxes_colors = [[0, 0, 255], [0, 0, 255], [0, 0, 255]]

for i in range(3):
    image_augmented = random_augmentations(image, config)

    image_augmented_encoded = tf.image.encode_png(image_augmented)
    tf.io.write_file(f"images/results/ximilar-similar_{i + 1}.png", image_augmented_encoded)

for i in range(3):
    image_augmented, bboxes_augmented = random_augmentations(image, config, bboxes=bboxes)

    image_augmented = tf.image.draw_bounding_boxes([image_augmented], [bboxes_augmented], bboxes_colors)[0]
    image_augmented = tf.cast(image_augmented, tf.uint8)  # for some reason, draw_bounding_boxes converts image to float

    image_augmented_encoded = tf.image.encode_png(image_augmented)
    tf.io.write_file(f"images/results/ximilar-similar_bboxes_{i + 1}.png", image_augmented_encoded)
