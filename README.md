# tf-image

__tf-image__ implements methods for image augmentation for Tensorflow 2.+ / tf.data.Dataset.  

__Why?__

Official TensorFlow 2+ [tf.image](https://www.tensorflow.org/api_docs/python/tf/image) package contains just a few and simple operations for image augmentation. This is not enough if you want to augment images and using all power from tf.data.Dataset. There is also [tf-addons](https://www.tensorflow.org/addons) projects which contains more of the operations (for example rotate).

If you do not require the operations in graph then simply use cv2, [imgaug](https://github.com/aleju/imgaug) or [albumentations](https://github.com/albumentations-team/albumentations) with `tf.py_function`. They have much more operations and options for image augmentation.

Feel free to improve and add more functions. Please only plain tensorflow2+, no opencv.

## Installation

__tf-image__ is not available on PyPi right now. For installation, clone the repository and install it as pip install -e ./ ...

## Quickstart

Here is simple example how to use some of the operations when working with `tf.data.Dataset`:

```python
import tensorflow as tf
import tensorflow_addons as tfa

from tf_image.core.random import random_function
from tf_image.core.ops import rgb_shift, channel_swap, channel_drop


def map_function(image_file, label):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    # in order to use tf_image safely, we need to convert 
    # image to [0, 1.0] range
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])

    # use tensorflow library
    image = tf.image.random_flip_left_right(image, seed=None)
    image = tf.image.random_flip_up_down(image, seed=None)

    # use tf_image library
    image = random_function(image, rgb_shift, 0.1, None, **{"r_shift": 0.1, "g_shift": 0.1, "b_shift": 0.1}) # do rgb shift with 10 % prob
    image = random_function(image, channel_swap, 0.1, None)
    image = channel_drop(image)

    # use tensorflow addons library
    image = tfa.image.rotate(image, 10)

    # convert image back to [0, 255] range if needed
    return image * 255, label


def return_dataset(self, image_files, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_files, labels))
    dataset = dataset.map(map_function)
    dataset = dataset.cache().repeat(10)
    return dataset.shuffle(len(image_files)).batch(20)
```

## Supported operations

Image augmentations:

* rgb_shift
* channel_drop
* channel_swap
* grayscale
* gaussian_noise

Random operations:

* random_function: calls function with some probability [0, 0.1]

Feel free to open merge requests and add more!

[![](logo.png)](https://ximilar.com)
