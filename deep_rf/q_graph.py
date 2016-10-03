import tensorflow as tf
import _utils


class QGraph(object):
    """Stores input and output Tensors of Graph representing Q-function.

    QGraph constructor takes two sets of parameters:
        1. If defining own Graph structure, provide q_input and q_output
        2. If using predefined Graph, provide name and necessary inputs

    Parameters
    ----------
    q_input : tf.placeholder, shape = [None, frame_height, frame_width, num_frames]
        Tensor that holds a batch_size of input game States for training.

    q_output: tf.Tensor, shape = [None, num_actions]
        Tensor of batch_size output vectors containing Q-values for each action.

    name : str, 'snake_default'
        Specify the name of a predefined Tensorflow Graph structure.

    frame_height : int
        For predefined Graph

    frame_width : int
        For predefined Graph

    num_frames : int
        For predefined Graph

    num_actions: int
        For predefined Graph


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

    def __init__(self, **kwargs):
        if 'name' in kwargs:
            name = kwargs['name']
            frame_height = kwargs['frame_height']
            frame_width = kwargs['frame_width']
            num_frames = kwargs['num_frames']
            num_actions = kwargs['num_actions']

            if name == 'snake_default':
                self.q_input, self.q_output = QGraph.create_snake_default(
                    frame_height, frame_width, num_frames, num_actions)
            else:
                raise Exception('No predefined QGraph with this name.')

        elif 'q_input' in kwargs and 'q_output' in kwargs:
            self.q_input = kwargs['q_input']
            self.q_output = kwargs['q_output']

        else:
            raise Exception(
                'Either provide q_input & q_output or choose a predefined Graph.')

        self.graph = self.q_input.graph
        self.var_list = self.graph.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES)

    @staticmethod
    def create_snake_default(frame_height, frame_width, num_frames,
                             num_actions):
        """3 conv + 2 fc layers. Using 16->8->8 2x2 kernels."""
        g = tf.Graph()

        with g.as_default():
            # input layer
            q_input = tf.placeholder(dtype=tf.float32,
                                     shape=[None, frame_height,
                                            frame_width, num_frames])

            # hidden convolutional layer 1
            w_conv1 = _utils.init_conv_filter(filter_height=2,
                                              filter_width=2,
                                              in_channels=int(
                                                  q_input.get_shape()[-1]),
                                              out_channels=16)
            h_conv1 = tf.nn.relu(_utils.conv2d(input=q_input,
                                               filter=w_conv1,
                                               stride=1))

            # hidden convolutional layer 2
            w_conv2 = _utils.init_conv_filter(1, 1,
                                              int(w_conv1.get_shape()[-1]), 8)
            h_conv2 = tf.nn.relu(_utils.conv2d(h_conv1, w_conv2, 1))

            # hidden convolutional layer 3
            w_conv3 = _utils.init_conv_filter(1, 1,
                                              int(w_conv2.get_shape()[-1]), 8)
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
                                           width=num_actions)
            b_fc2 = _utils.init_fc_bias(length=num_actions)
            q_output = tf.matmul(h_fc1, w_fc2) + b_fc2

        return q_input, q_output
