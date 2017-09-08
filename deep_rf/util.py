#
# Utility functions for defining Q graphs
#
# Authors:
#    Christopher Aicher <aicherc@uw.edu>
#    Luca Weihs <lucaw@uw.edu>
#    Kyle Lo <kyleclo@uw.edu>
#    Wesley Lee <wtlee@uw.edu>
#
# License: BSD 3 clause
#

import tensorflow as tf
from operator import mul


def init_conv_filter(filter_height, filter_width, in_channels, out_channels):
    """Initialize weights for Convolutional filter"""

    initial = tf.truncated_normal(shape=[filter_height, filter_width,
                                         in_channels, out_channels],
                                  stddev=0.1, mean=0.0)
    return tf.Variable(initial)


def conv2d(input, filter, stride):
    """Convolve Input and Filter Tensors without zero-padding"""

    return tf.nn.conv2d(input=input, filter=filter,
                        strides=[1, stride, stride, 1],
                        padding='VALID')


def flatten_4d_to_2d(input):
    """Reshapes a 4D Tensor to a 2D Tensor (preserves 1st dimension)"""
    num_cols = int(reduce(mul, input.get_shape()[1:]))
    return tf.reshape(input, [-1, num_cols])


def init_fc_weights(height, width):
    """Initialize weights to multiply against a Fully Connected layer"""

    initial = tf.truncated_normal(shape=[height, width],
                                  stddev=0.1, mean=0.0)
    return tf.Variable(initial)


def init_fc_bias(length):
    """Initialize bias to add to a Fully Connected layer"""

    initial = tf.constant(0.01, shape=[length])
    return tf.Variable(initial)



