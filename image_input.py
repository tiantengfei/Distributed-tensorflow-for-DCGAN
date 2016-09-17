from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


IMAGE_SIZE = 64

NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 4000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1770
batch_size = 64


def read_cifar10(filename_queue):
    """ Read a example from the filename queue. The TFRecordReader is used to
     read examples from tfrecords' files. The decoder of decode_raw is used to
     decode the tf.string of the example.

     Args:
         filename_queue: filename's queue where the file reader read from will be placed.

     Returns:
         A example used to train.
    """

    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()
    label_dim = 300
    result.height = 64
    result.width = 64
    result.depth = 3

    reader = tf.TFRecordReader()
    _, serializded_example = reader.read(filename_queue)
    features = tf.parse_single_example(serializded_example,
                                       features={
                                           'desc_vector': tf.FixedLenFeature([], tf.string),
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.string)
                                       })
    image = tf.decode_raw(features['image_raw'], tf.uint8)

    depth_major = tf.reshape(image,
                             [result.height, result.width, result.depth])

    result.uint8image = depth_major  # tf.transpose(depth_major, [1, 2, 0])

    desc_vector = tf.decode_raw(features['desc_vector'], tf.float32)
    result.desc_vector = tf.reshape(desc_vector, [300])

    label = tf.decode_raw(features['label'], tf.float32)
    result.label = tf.reshape(label, [1])
    return result


def _generate_image_and_label_batch(image, desc_vector, label, min_queue_examples,
                                    batch_size, shuffle):
    """ generate a batch of images and labels.

    Args:
        image: the trained image.
        label: label correspond to the image.
        min_queue_examples: the least examples int the example's queue.
        batch_size: the size of a batch.
        shuffle: whether or not to shuffle the examples.

    Returns:
        A batch of examples including images and the corresponding label.
    """
    num_preprocess_threads = 16
    if shuffle:
        images, desc_vector, label_batch = tf.train.shuffle_batch(
            [image, desc_vector, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, desc_vector, label_batch = tf.train.batch(
            [image, desc_vector, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)


    # Display the training images in the visualizer.
    #tf.image_summary('images', images)
    return images, desc_vector, label_batch


def distorted_inputs(data_dir, batch_size=64):
    """ distort the images and get a batch of trained images.
    Args:
        data_dir: directory that place the images' data.
        batch_size: the number of images that a step will be trained.

    Returns:
        A batch of examples including images and the corresponding label.
    """
    filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    #filenames = [os.path.join(data_dir, 'image_%d.tfrecords' % i)
    #              for i in xrange(0, 1)]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = 50
    width = 50

    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    distorted_image = tf.image.random_flip_left_right(distorted_image)

    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    float_image = tf.image.per_image_whitening(distorted_image)

    min_fraction_of_examples_in_queue = 0.25
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.desc_vector, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


