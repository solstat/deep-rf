import tensorflow as tf
import numpy as np
from operator import mul

class QGraph(object):
    """
        q_input: (tf.placeholder float [None, board_height, board_width, num_frames]) - tf placeholder for state
        q_output: (tf.Tensor of action_values [None, num_actions]) - Q function output to evaluated with tf.run()

    """
    def __init__(self, q_input, q_output):
        self.q_input = q_input
        self.q_output = q_output
        self.graph = self.q_input.graph
        self.var_list = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


    @classmethod
    def default_q_graph(cls, game, num_frames):
        """ initialize Q function input & output

            Returns:
                QGraph: a q graph
        """

        g = tf.Graph()

        with g.as_default():
            q_input = tf.placeholder(tf.float32, [None, game.frame_height,
                                                  game.frame_width,
                                                  num_frames])

            w_conv1 = cls._filter_variable(2, 2,
                                           in_channels=int(q_input.get_shape()[-1]),
                                           out_channels=16)
            h_conv1 = tf.nn.relu(cls._conv2d(q_input, w_conv1, stride=1))

            w_conv2 = cls._filter_variable(1, 1,
                                           in_channels=int(w_conv1.get_shape()[-1]),
                                           out_channels=8)
            h_conv2 = tf.nn.relu(cls._conv2d(h_conv1, w_conv2, stride=1))

            w_conv3 = cls._filter_variable(1, 1,
                                           in_channels=int(w_conv2.get_shape()[-1]),
                                           out_channels=8)
            h_conv3 = tf.nn.relu(cls._conv2d(h_conv2, w_conv3, stride=1))

            dim_conv3 = int(reduce(mul, h_conv3.get_shape()[1:]))
            q_state_flat = tf.reshape(h_conv3, [-1, dim_conv3])
            w_fc1 = cls._matmul_variable(dim_conv3, dim_conv3)
            b_fc1 = cls._bias_variable(dim_conv3)
            h_fc1 = tf.nn.relu(tf.matmul(q_state_flat, w_fc1) + b_fc1)

            w_out = cls._matmul_variable(dim_conv3, len(game.action_list))
            b_out = cls._bias_variable(len(game.action_list))
            q_output = tf.matmul(h_fc1, w_out) + b_out

        return QGraph(q_input, q_output)


    @staticmethod
    def _filter_variable(filter_height, filter_width, in_channels, out_channels):
        initial = tf.truncated_normal(shape=[filter_height, filter_width, in_channels, out_channels],
                                      stddev=0.1, mean=0.0)
        return tf.Variable(initial)


    @staticmethod
    def _matmul_variable(height, width):
        initial = tf.truncated_normal(shape=[height, width], stddev=0.1, mean=0.0)
        return tf.Variable(initial)


    @staticmethod
    def _bias_variable(out_channels):
        initial = tf.constant(0.01, shape=[out_channels])
        return tf.Variable(initial)


    @staticmethod
    def _conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')