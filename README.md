# tf-image

__tf-image__ implements methods for image augmentation for Tensorflow 2.+ / tf.data.Dataset.  

__Why?__

Official TensorFlow 2+ [tf.image](https://www.tensorflow.org/api_docs/python/tf/image) package contains just 
a few and simple operations for image augmentation. This is not enough if you want to augment images and using 
all the power of tf.data.Dataset. There is also [tf-addons](https://www.tensorflow.org/addons) projects which 
contains more of the operations (for example rotate), but it still not enough.
And on top of that, none of those two supports operation on bounding boxes and therefore is not fully usable 
for augmentation object detection datasets.

If you do not require the operations in graph then simply use cv2, [imgaug](https://github.com/aleju/imgaug) 
or [albumentations](https://github.com/albumentations-team/albumentations) together with `tf.py_function`. 
They have (at the moment) much more operations and options for image augmentation.

## Installation

Use pip:

    pip install tf-image

For installation from source code, clone the repository and install it from code (`pip install -e .`).
There are no dependencies specified. You have to install TensorFlow 2+ and appropriate TensorFlow Addons.
Specific version is on you, we wanted to keep this library as general as possible.

## Image and bounding boxes formats
We use channel last format for images. Images can be represented either in 0.0 - 1.0 or 0 - 255 range.
Similar is true for bounding boxes. They can be provided either in relative coordinates with range 0.0 - 1.0 using
float dtype or in absolute image coordinates using integer dtype.
Internally, This is done using [convert_type](tf_image/core/convert_type_decorator.py) 
decorator on functions which needs it. This decorator converts the images into the type we use 
(tf.float and 0.0-1.1 in both cases) and after the function is done, original format is restored. 
If performing multiple operations, you can use this decorator on own function.
(Conversions after each operation will not be needed.)

## Quickstart
For your convenience, we included a simple and configurable application, which combines all the provided augmentations.
They are performed in a random order to make the augmentation even more powerful.

There is also one script which uses this augmentation function and which outputs three augmented 
image without bounding boxes and three with bonding boxes. 
See [example/example.py](example/example.py) for more information.

If you want to use the functions alone, here is how:
```python
import tensorflow as tf
import tensorflow_addons as tfa

from tf_image.core.random import random_function
from tf_image.core.colors import rgb_shift, channel_drop
from tf_image.core.convert_type_decorator import convert_type


@convert_type
def augment_image(image):
    # use TensorFlow library
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # use tf-image library
    image = random_function(
        image, rgb_shift, 0.1, None, **{"r_shift": 0.1, "g_shift": 0.1, "b_shift": 0.1}
    )  # do rgb shift with 10 % prob
    image = random_function(image, channel_drop, 0.1, None)
    # and whatever else you want

    # use TensorFlow Addons library
    image = tfa.image.rotate(image, 10)

    return image


def map_function(image_file, label):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = augment_image(image)

    return image, label


def return_dataset(image_files, labels):
    dataset = (
        tf.data.Dataset.from_tensor_slices((image_files, labels))
        .cache()
        .shuffle(len(image_files))
        .map(map_function)
        .batch(20)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    return dataset

return_dataset(["images/ximilar-similar.jpg"], [[1,2,3]])
```

## Supported operations

Image augmentations:
* aspect ration deformations *(inc. bounding boxes)*
* channel drop
* channel swap
* erase, see [https://arxiv.org/abs/1708.04552]  *(repeated, inc. bounding boxes)*
* flip up-down, left-right *(inc. bounding boxes)*
* grayscale
* gaussian noise
* clip *(inc. bounding boxes)*
* rgb shift
* resize with different methods *(inc. bounding boxes)*
* rotate *(inc. bounding boxes)*

Random operations:
* random_function: calls function on image with some probability [0.0, 0.1]
* random_function_bboxes: calls function on image and bounding boxes with some probability [0.0, 0.1]

Feel free to improve and add more functions. We are looking forward to your merge requests!
(Please only plain tensorflow2+, no opencv.)

[![](logo.png)](https://ximilar.com)
