import tensorflow as tf
import _utils


class QGraph(object):
    """ Tensorflow Graph for the Q Function

        Parameters:
            q_input (tf.placeholder float [None, board_height, board_width, num_frames]):
                tf placeholder for state
            q_output (tf.Tensor of action_values [None, num_actions]):
                Q function output to evaluated with tf.run()

        Attributes:
            q_input
            q_output
            graph
            var_list

    """
    def __init__(self, q_input, q_output):
        self.q_input = q_input
        self.q_output = q_output
        self.graph = self.q_input.graph
        self.var_list = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


    @staticmethod
    def default_q_graph(game, num_frames):
        """ initialize Q function input & output

            Parameters:
                game (SinglePlayerGame): a game object
                num_frames (int): number of past frames to keep in memory

            Returns:
                QGraph: a q graph
        """

        g = tf.Graph()

        with g.as_default():
            # input layer
            q_input = tf.placeholder(dtype=tf.float32,
                                     shape=[None, game.frame_height,
                                            game.frame_width, num_frames])

            # hidden convolutional layer 1
            w_conv1 = _utils.init_conv_filter(filter_height=2,
                                              filter_width=2,
                                              in_channels=int(q_input.get_shape()[-1]),
                                              out_channels=16)
            h_conv1 = tf.nn.relu(_utils.conv2d(input=q_input,
                                               filter=w_conv1,
                                               stride=1))

            # hidden convolutional layer 2
            w_conv2 = _utils.init_conv_filter(1, 1, int(w_conv1.get_shape()[-1]), 8)
            h_conv2 = tf.nn.relu(_utils.conv2d(h_conv1, w_conv2, 1))

            # hidden convolutional layer 3
            w_conv3 = _utils.init_conv_filter(1, 1, int(w_conv2.get_shape()[-1]), 8)
            h_conv3 = tf.nn.relu(_utils.conv2d(h_conv2, w_conv3, 1))

            # reshape convolutional layer from 4D -> 2D tensor
            h_conv3_flat = _utils.flatten_4d_to_2d(h_conv3)
            len_h_conv3 = int(h_conv3_flat.get_shape()[-1])

            # hidden fully connected layer 1
            w_fc1 = _utils.init_fc_weights(height=len_h_conv3,
                                           width=len_h_conv3)
            b_fc1 = _utils.init_fc_bias(length=len_h_conv3)
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, w_fc1) + b_fc1)

            # output = fully connected layer 2
            w_fc2 = _utils.init_fc_weights(height=len_h_conv3,
                                           width=len(game.action_list))
            b_fc2 = _utils.init_fc_bias(length=len(game.action_list))
            q_output = tf.matmul(h_fc1, w_fc2) + b_fc2

        return QGraph(q_input, q_output)
