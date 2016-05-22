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

    def __init__(self, board_height, board_width):
        self.epsilon = 1
        self.board_width = board_width
        self.board_height = board_height
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
        return None

    def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def _bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
