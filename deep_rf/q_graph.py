#
# Abstract base class for SinglePlayerGames.
# For deep_rf learners to play well with a provided game, recommended
# extending this class and implementing the specified methods.
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
import util


class QGraph(object):
    """Stores input and output Tensors of Graph representing Q-function.

    Parameters
    ----------
    q_input : tf.placeholder, shape = [None, frame_height, frame_width, num_frames]
        Tensor that holds a batch_size of input game States for training.

    q_output: tf.Tensor, shape = [None, num_actions]
        Tensor of batch_size output vectors containing Q-values for each action.

    Attributes
    ----------
    q_input : tf.placeholder
        See above.

    q_output : tf.Tensor
        See above.

    graph : tf.Graph
        Tensorflow graph corresponding to q_input.

    var_list : list of str
        Contains names of trainable variables in Graph.
    """

    def __init__(self, q_input, q_output):
        self.q_input = q_input
        self.q_output = q_output
        self.graph = self.q_input.graph
        self.var_list = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    @staticmethod
    def create_3conv2fc(game, num_frames, params):
        """3 convolutional and 2 fully connected layers w/ ReLu activation

        Parameters
        ----------
        game: SinglePlayerGame

        num_frames : int

        params: dict

            Keywords:
                'filter': list of 2 ints [kernel_height, kernel_width],
                'out' int representing number of out_channels of kernel,
                'stride' int representing number of kernel strides

            Example:
                params = {
                    'filter1': [2, 2], 'out1': 16, 'stride1': 1,
                    'filter2': [1, 1], 'out2': 8, 'stride2': 1,
                    'filter3': [1, 1], 'out3': 8, 'stride3': 1,
                    'out4':
                }
        """

        frame_height = game.frame_height
        frame_width = game.frame_width
        num_actions = len(game.action_list)

        g = tf.Graph()
        with g.as_default():

            # input layer
            q_input = tf.placeholder(dtype=tf.float32,
                                     shape=[None,
                                            frame_height,
                                            frame_width,
                                            num_frames])

            # hidden convolutional layer 1
            w_conv1 = util.init_conv_filter(filter_height=params['filter1'][0],
                                            filter_width=params['filter1'][1],
                                            in_channels=int(q_input.get_shape()[-1]),
                                            out_channels=params['out1'])
            h_conv1 = tf.nn.relu(util.conv2d(input=q_input,
                                             filter=w_conv1,
                                             stride=params['stride1']))

            # hidden convolutional layer 2
            w_conv2 = util.init_conv_filter(filter_height=params['filter2'][0],
                                            filter_width=params['filter2'][1],
                                            in_channels=int(w_conv1.get_shape()[-1]),
                                            out_channels=params['out2'])
            h_conv2 = tf.nn.relu(util.conv2d(input=h_conv1,
                                             filter=w_conv2,
                                             stride=params['stride2']))

            # hidden convolutional layer 3
            w_conv3 = util.init_conv_filter(filter_height=params['filter3'][0],
                                            filter_width=params['filter3'][1],
                                            in_channels=int(w_conv2.get_shape()[-1]),
                                            out_channels=params['out3'])
            h_conv3 = tf.nn.relu(util.conv2d(input=h_conv2,
                                             filter=w_conv3,
                                             stride=params['stride3']))

            # reshape convolutional layer from 4D -> 2D tensor
            h_conv3_flat = util.flatten_4d_to_2d(h_conv3)
            len_h_conv3 = int(h_conv3_flat.get_shape()[-1])

            # hidden fully connected layer 1
            w_fc1 = util.init_fc_weights(height=len_h_conv3,
                                         width=len_h_conv3)
            b_fc1 = util.init_fc_bias(length=len_h_conv3)
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, w_fc1) + b_fc1)

            # output = fully connected layer 2
            w_fc2 = util.init_fc_weights(height=len_h_conv3,
                                         width=num_actions)
            b_fc2 = util.init_fc_bias(length=num_actions)
            q_output = tf.matmul(h_fc1, w_fc2) + b_fc2

        q_graph = QGraph(q_input, q_output)
        return q_graph

