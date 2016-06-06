#!/usr/bin/env python
"""
Does Deep Q-Learning for Snake

"""

# Import Modules
import numpy as np
import tensorflow as tf
from board_state import BoardState, ACTION

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
        frames_tuple (T tuple of  board_height by board_width ndarrays)

    Methods:
        new_state_from_old(new_frame) - return new State object
        to_array() - return (T by board_height by board_width ndarray) representation
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

    def learn_q_function(self):
        # Init Q network
        # Optional Load Saved Q parameter

        # For Training Time
            # Get next sample -> List of Batches
            # Get a minibatch -> partially Optimize Q for loss

        # Return Q or Q parameters

    def get_next_sample(self):
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
        # Unimplemented
        raise Exception("Unimplemented")

    def _init_network(self):
        """ initialize Q function input & output

        Returns:
            q_state: (tf.placeholder float [None, state_size]) - tf placeholder for state
            q_output: (tf.Tensor of action_values) - Q function output to evaluated with tf.run()

        """
        frame_size = self.board_height * self.board_width
        state_size = frame_size * self.num_frames
        q_state = tf.placeholder(float, [None, self.board_height, self.board_width, self.num_frames])

        weights1 = self._weight_variable([8, 8, self.num_frames, 32])
        biases1 = self._bias_variable([32])
        conv1 = tf.nn.relu(self._conv2d(q_state, weights1, stride=4) + biases1)

        weights2= self._weight_variable([4, 4, 32, 64])
        biases2 = self._bias_variable([64])
        conv2 = tf.nn.relu(self._conv2d(conv1, weights2, stride=2) + biases2)

        weights3 = self._weight_variable([3, 3, 64, 64])
        biases3 = self._bias_variable([64])
        conv3 = tf.nn.relu(self._conv2d(conv2, weights3, stride=1) + biases3)

        height_after = self.board_height / 4 / 2 / 1 # Dimension Size decreases by conv strides
        width_after = self.board_width / 4 / 2 / 1   # Dimension Size decreases by conv strides
        conv3_flat = tf.reshape(conv3, [-1, height_after * width_after * 64])

        weights4 = self._weight_variable([height_after * width_after * 64, height_after * width_after * 16])
        biases4 = self._bias_variable([height_after * width_after * 16])
        fc1 = tf.nn.relu(tf.matmul(conv3_flat, weights4) + biases4)

        weights5 = self._weight_variable([height_after * width_after * 16, height_after * width_after * 4])
        biases5 = self._bias_variable([height_after * width_after * 4])
        fc2 = tf.nn.relu(tf.matmul(fc1, weights5) + biases5)

        #action_q_values = tf.placeholder("float", [None, 4]) # 4 == num actions
        weights6 = self._weight_variable([height_after * width_after * 4, 4])
        biases6 = self._bias_variable([4])
        q_output = tf.nn.softmax(tf.matmul(fc2, weights6) + biases6)

        return q_state, q_output




    def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def _bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def _conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
