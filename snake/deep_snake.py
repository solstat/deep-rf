#!/usr/bin/env python
"""
Does Deep Q-Learning for Snake

"""

# Import Modules
import numpy as np
import tensorflow as tf
from board_state import BoardState, ACTION, ACTION_LIST

class ExperienceTuple:
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

class State:
    """
    State object for Q-learning
    Tuple of frames from snake game

    Args:
        frames_tuple (num_frames tuple of  board_height by board_width ndarrays)

    Methods:
        new_state_from_old(new_frame) - return new State object
        to_array() - return (board_height by board_width by num_frames ndarray) representation
    """
    def __init__(self, frames_tuple):
        self.frames_tuple = frames_tuple

    def new_state_from_old(self, new_frame):
        return State(self.frames_tuple[1:] + (new_frame,))

    def to_array(self):
        return np.dstack(self.frames_tuple)

class DeepSnake(object):

    def __init__(self, board_height, board_width, num_frames=2):
        self.epsilon = 1
        self.board_width = board_width
        self.board_height = board_height
        self.num_frames = num_frames
        self.time_penalty = -.01
        self.death_penalty = -1.0
        self.the_board = BoardState(self.board_height, self.board_width)
        self.experience_replay = []
        self._q_state, self._q_output = self._init_network()
        self._y_obs, self._action_indices, self._loss = self._init_training_loss_operation()
        self._optimizer = tf.train.AdamOptimizer().minimize(self._loss)
        self._sess = tf.Session()
        self._sess.run(tf.initialize_all_variables())


    def __del__(self):
        self._sess.close()


    def learn_q_function(self):
        # For Training Time
            # Get next sample -> List of ExperienceTuples
            # Get a minibatch -> partially Optimize Q for loss

        # Return Q or Q parameters

        for it in xrange(1000):
            if it % 100 == 0:
                print it
            for __ in xrange(50):
                self.experience_replay.append(self.get_next_experience_tuple())
            experience_batch = np.random.choice(self.experience_replay, 50, replace=False)
            actions_batch = [et.action for et in experience_batch] # TODO: don't use this as an index
            states_batch = [et.state.to_array() for et in experience_batch]
            y_targets = self._get_target_values(experience_batch)

            for __ in xrange(50):
                self._sess.run(self._optimizer(), feed_dict={self._q_state: states_batch,
                                                             self._action_indices: actions_batch,
                                                             self._y_obs: y_targets})
        return


    def get_next_experience_tuple(self):
        """ Yield the Experience Tuple for training Q
        yields:
          experience_tuple (Experience Tuple) - current state, action, reward, new_state
        """
        while True:
            self.the_board.reset()
            first_frame = self.the_board.get_frame()
            state_padding = [np.zeros(first_frame.shape) for _ in range(self.num_frames - 1)]
            current_state = State(tuple(state_padding) + (first_frame,))

            while not self.the_board.is_game_over():
                action = self.get_action_for_state(current_state)
                last_score = self.the_board.get_score()
                self.the_board.do_action(action)
                new_state = current_state.new_state_from_old(self.the_board.get_frame())
                new_score = self.the_board.get_score()
                reward = self.calculate_reward(last_score, new_score)
                yield ExperienceTuple(current_state, action, reward, new_state)
                current_state = new_state


    def calculate_reward(self, last_score, new_score):
        """ Define a reward based on score
        Args:
          last_score (int) - score of current state
          new_score (int) - score of new state
        Returns:
          reward (double) - reward
        """
        if self.the_board.is_game_over():
            reward = self.death_penalty
        elif new_score == last_score:
            reward = self.time_penalty
        else:
            reward = float(new_score - last_score)
        return reward


    def get_action_for_state(self, state):
        if self.epsilon >= np.random.rand():
            return np.random.choice(ACTION_LIST)
        else:
            q_values = self._sess.run(self._q_output, feed_dict={self._q_state: state})
            return ACTION_LIST[np.argmax(q_values)]



    def _init_training_loss_operation(self):
        y_obs = tf.placeholder(dtype=tf.float32, shape=[None])
        action_indices = tf.placeholder(dtype=tf.int16, shape=[None])
        loss = tf.reduce_mean((y_obs - self._q_output[:,action_indices])^2)
        return y_obs, action_indices, loss


    def _get_target_values(self, experience_batch):
        """
        Args:
            experience_batch:  list of ExperienceTuples

        Returns:
            y_target:   np.ndarray of [batch_size, r + max Q(s')]
        """
        rewards = np.array([et.reward for et in experience_batch])
        states = [et.next_state.to_array() for et in experience_batch]
        q_values = self._sess.run(self._q_output, feed_dict={self._q_state: states})
        y_target = rewards + np.max(q_values, axis=1)
        return y_target



    def _init_network(self):
        """ initialize Q function input & output

        Returns:
            q_state: (tf.placeholder float [None, board_height, board_width, num_frames]) - tf placeholder for state
            q_output: (tf.Tensor of action_values [None, 4]) - Q function output to evaluated with tf.run()

        """
        q_state = tf.placeholder(float, [None, self.board_height, self.board_width, self.num_frames])

        weights1 = self._filter_variable(filter_height=8, filter_width=8, in_channels=self.num_frames, out_channels=32)
        biases1 = self._bias_variable(out_channels=32)
        conv1 = tf.nn.relu(self._conv2d(q_state, weights1, stride=4) + biases1)
        # q_state is [batch_size, height, width, in_channels]
        # weights is [filter_height, filter_width, in_channels, out_channels]
        # conv is [batch_size, height/stride, width/stride, out_channels]

        weights2= self._filter_variable(4, 4, 32, 64)
        biases2 = self._bias_variable(64)
        conv2 = tf.nn.relu(self._conv2d(conv1, weights2, stride=2) + biases2)

        weights3 = self._filter_variable(3, 3, 64, 64)
        biases3 = self._bias_variable(64)
        conv3 = tf.nn.relu(self._conv2d(conv2, weights3, stride=1) + biases3)

        height_after = self.board_height / 4 / 2 / 1 # Dimension Size decreases by conv strides
        width_after = self.board_width / 4 / 2 / 1   # Dimension Size decreases by conv strides
        conv3_flat = tf.reshape(conv3, [-1, height_after * width_after * 64])

        weights4 = self._matmul_variable(height_after * width_after * 64, height_after * width_after * 16)
        biases4 = self._bias_variable(height_after * width_after * 16)
        fc1 = tf.nn.relu(tf.matmul(conv3_flat, weights4) + biases4)

        weights5 = self._matmul_variable(height_after * width_after * 16, height_after * width_after * 4)
        biases5 = self._bias_variable(height_after * width_after * 4)
        fc2 = tf.nn.relu(tf.matmul(fc1, weights5) + biases5)

        #action_q_values = tf.placeholder("float", [None, 4]) # 4 == num actions
        weights6 = self._matmul_variable(height_after * width_after * 4, 4)
        biases6 = self._bias_variable(4)
        q_output = tf.matmul(fc2, weights6) + biases6

        return q_state, q_output


    @staticmethod
    def _filter_variable(filter_height, filter_width, in_channels, out_channels):
        initial = tf.truncated_normal(shape=[filter_height, filter_width, in_channels, out_channels],
                                      stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def _matmul_variable(height, width):
        initial = tf.truncated_normal(shape=[height, width], stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def _bias_variable(out_channels):
        initial = tf.constant(0.1, shape=[out_channels])
        return tf.Variable(initial)

    @staticmethod
    def _conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
