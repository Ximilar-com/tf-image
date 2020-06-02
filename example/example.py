from PIL import Image
import numpy as np
import tensorflow as tf

from tf_image.application.augmentation_config import AugmentationConfig
from tf_image.application.tools import apply_random_order_augmentations

#
# Loads an images from images/ximilar-similar.jpg, create some bounding boxes and augment
# three times without bounding boxes and three times with them. Results are saved to images/results folder.
#

# Loads the basic setup, feel free to experiment!
config = AugmentationConfig()

# Loads the image and creates bounding boxes for three completely visible apples.
image = np.array(Image.open("images/ximilar-similar.jpg"))
bboxes = np.array([[262.0, 135.0, 504.0, 371.0], [285.0, 446.0, 494.0, 644.0], [272.0, 688.0, 493.0, 895.0]])
bboxes /= [image.shape[0], image.shape[1], image.shape[0], image.shape[1]]

for i in range(3):
    image_augmented = apply_random_order_augmentations(np.copy(image), config)
    image_augmented = Image.fromarray(image_augmented.numpy())
    image_augmented.save(f"images/results/ximilar-similar_{i + 1}.png")

for i in range(3):
    image_augmented, bboxes_augmented = apply_random_order_augmentations(np.copy(image), config, bboxes=bboxes)
    image_augmented = tf.image.draw_bounding_boxes([image_augmented], [bboxes_augmented], [(0, 0, 255)] * len(bboxes))
    image_augmented = tf.cast(image_augmented[0], dtype=tf.uint8).numpy()
    image_augmented = Image.fromarray(image_augmented)
    image_augmented.save(f"images/results/ximilar-similar_bboxes_{i + 1}.png")
