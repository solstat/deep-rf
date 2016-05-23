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
    def __init__(self, frames_tuple):
        self.frames_tuple = frames_tuple

    def new_state_from_old(self, new_frame):
        return State(self.frames_tuple[1:] + (new_frame,))

    def to_array(self):
        return np.dstack(self.frames_tuple)

class DeepSnake(object):

    def __init__(self, board_height, board_width, num_frames):
        self.epsilon = 1
        self.board_width = board_width
        self.board_height = board_height
        self.num_frames = num_frames
        self.time_penalty = -.01
        self.death_penalty = -1.0
        self.the_board = BoardState(self.board_height, self.board_width)

    def get_next_sample(self):
        while True:
            self.the_board.reset()
            first_frame = self.the_board.get_frame()
            current_state = State((first_frame * 0, first_frame))

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
        if self.the_board.is_game_over():
            return self.death_penalty
        elif new_score == last_score:
            return self.time_penalty
        else:
            return new_score - last_score

    def get_action_for_state(self, state):
        # Unimplemented
        raise Exception("Unimplemented")

    def _init_network(self):
        frame_size = self.board_height * self.board_width
        state_size = frame_size * self.num_frames
        q_state = tf.placeholder("float", [None, state_size])

        weights1 = self._weight_variable([8, 8, 2, 32])
        biases1 = self._bias_variable([32])
        conv1 = tf.nn.relu(self._conv2d(q_state, weights1, 4) + biases1)

        weights2= self._weight_variable([4, 4, 32, 64])
        biases2 = self._bias_variable([64])
        conv2 = tf.nn.relu(self._conv2d(conv1, weights2, 2) + biases2)

        weights3 = self._weight_variable([3, 3, 64, 64])
        biases3 = self._bias_variable([64])
        conv3 = tf.nn.relu(self._conv2d(conv2, weights3, 1) + biases3)

        height_after = self.board_height / 4 / 2
        width_after = self.board_width / 4 / 2
        weights4 = self._weight_variable([height_after * width_after * 64])
        biases4 = self._bias_variable()
        conv3_flat = tf.reshape(conv3, [-1, height_after * width_after * 64])
        fc1 = tf.nn.relu(tf.matmul(conv3_flat, weights4) + biases4)

        weights5 = self._weight_variable([height_after * width_after * 64])


        action_q_values = tf.placeholder("float", [None, 4]) # 4 == num actions





    def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def _bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def _conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
