#!/usr/bin/env python
"""
Does Deep Q-Learning for Snake

"""

# Import Modules
import numpy as np
import tensorflow as tf
from board_state import BoardState, ACTION, ACTION_LIST
from ascii_snake import boardToString
from operator import mul
import os

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

    def __init__(self, board_height, board_width, num_frames=2, tf_file_save_path=None):
        self.epsilon = 1.0
        self.gamma = 0.9
        self.board_width = board_width
        self.board_height = board_height
        self.num_frames = num_frames
        self.death_penalty = -1.0
        self.time_penalty = -0.01
        self.the_board = BoardState(self.board_height, self.board_width)
        self.experience_replay = []
        self._tf_file_save_path = tf_file_save_path
        self._q_state, self._q_output = self._init_network()
        self._y_obs, self._action_indices, self._loss = self._init_training_loss_operation()
        self._optimizer = tf.train.AdamOptimizer().minimize(self._loss)
        self._saver = tf.train.Saver()
        self._sess = tf.Session()

        if tf_file_save_path is not None and os.path.exists(tf_file_save_path):
            self._saver.restore(self._sess, tf_file_save_path)
            print("Model restored from " + tf_file_save_path + ".")
        else:
            self._sess.run(tf.initialize_all_variables())


    def __del__(self):
        self._sess.close()


    def save_tf_weights(self):
        if self._tf_file_save_path is not None:
            self._saver.save(self._sess, self._tf_file_save_path)


    def learn_q_function(self, num_iterations=1000, batch_size=50,
                         num_training_steps=10, play_frequency=100,
                         epsilon_multiplier=1.0):
        # For Training Time
            # Get next sample -> List of ExperienceTuples
            # Get a minibatch -> partially Optimize Q for loss

        # Return Q or Q parameters

        experience_tuple_generator = self.get_next_experience_tuple()
        for it in xrange(num_iterations):
            if it != 0 and it % play_frequency == 0:
                print y_targets
                self.play_one_game(0.0)
            print it
            for __ in xrange(batch_size):
                self.experience_replay.append(experience_tuple_generator.next())
            experience_batch = np.random.choice(self.experience_replay, batch_size, replace=False)
            actions_batch = [et.action.value for et in experience_batch] # TODO: don't use this as an index
            states_batch = [et.state.to_array() for et in experience_batch]
            y_targets = self._get_target_values(experience_batch)

            for __ in xrange(num_training_steps):
                self._sess.run(self._optimizer, feed_dict={self._q_state: states_batch,
                                                             self._action_indices: actions_batch,
                                                             self._y_obs: y_targets})

            self.epsilon *= epsilon_multiplier

        return

    def play_one_game(self, my_epsilon):
        old_epsilon = self.epsilon
        self.epsilon = my_epsilon

        new_board = BoardState(self.board_height, self.board_width)
        first_frame = new_board.get_frame()
        state_padding = [np.zeros(first_frame.shape) for _ in range(self.num_frames - 1)]
        current_state = State(tuple(state_padding) + (first_frame,))

        while True:
            print "\n\n\n\n\n\n\n"
            print boardToString(new_board.get_frame().T)
            action = self.get_action_for_state(current_state)
            r = raw_input("Press q to quit or Press r to reset: ")
            if r == 'q':
                break
            elif r == 'r' or new_board.is_game_over():
                new_board = BoardState(self.board_height, self.board_width)
                first_frame = new_board.get_frame()
                state_padding = [np.zeros(first_frame.shape) for _ in
                                 range(self.num_frames - 1)]
                current_state = State(tuple(state_padding) + (first_frame,))
            else:
                new_board.do_action(action)
                current_state = current_state.new_state_from_old(new_board.get_frame())

        self.epsilon = old_epsilon
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
                if self.the_board.is_game_over():
                    yield ExperienceTuple(current_state, action, reward, None)
                else:
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
            #raise Exception("This was just to test without pellet")
            reward = float(new_score - last_score)
        return reward


    def get_action_for_state(self, state):
        if self.epsilon >= np.random.rand():
            return np.random.choice(ACTION_LIST)
        else:
            q_state = np.array([state.to_array()])
            q_values = self._sess.run(self._q_output, feed_dict={self._q_state: q_state})
            if self.epsilon == 0:
                print ACTION_LIST
                print q_values
                print ACTION_LIST[np.argmax(q_values)]
            return ACTION_LIST[np.argmax(q_values)]



    def _init_training_loss_operation(self):
        y_obs = tf.placeholder(dtype=tf.float32, shape=[None])
        action_indices = tf.placeholder(dtype=tf.int64, shape=[None])
        action_one_hot = tf.one_hot(action_indices, 4, on_value=1.0, off_value=0.0)
        y_pred = tf.reduce_sum(self._q_output * action_one_hot, reduction_indices=1)
        loss = tf.reduce_mean((y_obs - y_pred)**2)
        return y_obs, action_indices, loss


    def _get_target_values(self, experience_batch):
        """
        Args:
            experience_batch:  list of ExperienceTuples

        Returns:
            y_target:   np.ndarray of [batch_size, r + max Q(s')]
        """
        rewards = np.array([et.reward for et in experience_batch])
        states = [et.next_state.to_array() if et.next_state is not None else et.state.to_array() for et in experience_batch]
        q_values = self._sess.run(self._q_output, feed_dict={self._q_state: states})
        game_not_over_indicator = np.array([1.0 if et.next_state is not None else 0.0 for et in experience_batch])
        y_target = rewards + self.gamma * np.max(q_values, axis=1) * game_not_over_indicator
        return y_target



    def _init_network(self):
        """ initialize Q function input & output

        Returns:
            q_state: (tf.placeholder float [None, board_height, board_width, num_frames]) - tf placeholder for state
            q_output: (tf.Tensor of action_values [None, 4]) - Q function output to evaluated with tf.run()

        """
        # q_state = tf.placeholder(tf.float32, [None, self.board_height, self.board_width, self.num_frames])
        # q_state_flat = tf.reshape(q_state, [-1, self.board_height * self.board_width * self.num_frames])
        # w = self._matmul_variable(self.board_height * self.board_width * self.num_frames, 4)
        # #b = self._bias_variable(4)
        # q_output = tf.matmul(q_state_flat, w)# + b
        #
        # return q_state, q_output

        q_state = tf.placeholder(tf.float32, [None, self.board_height, self.board_width, self.num_frames])
        w_conv1 = self._filter_variable(2, 2, in_channels=self.num_frames, out_channels=8)
        h_conv1 = tf.nn.relu(self._conv2d(q_state, w_conv1, stride=1))

        w_conv2 = self._filter_variable(1, 1, in_channels=int(w_conv1.get_shape()[-1]), out_channels=8)
        h_conv2 = tf.nn.relu(self._conv2d(h_conv1, w_conv2, stride=1))

        w_conv3 = self._filter_variable(1, 1, in_channels=int(w_conv2.get_shape()[-1]), out_channels=8)
        h_conv3 = tf.nn.relu(self._conv2d(h_conv2, w_conv3, stride=1))

        dim_conv3 = int(reduce(mul, h_conv3.get_shape()[1:]))
        q_state_flat = tf.reshape(h_conv3, [-1, dim_conv3])
        w_fc1 = self._matmul_variable(dim_conv3, dim_conv3)
        b_fc1 = self._bias_variable(dim_conv3)
        h_fc1 = tf.nn.relu(tf.matmul(q_state_flat, w_fc1) + b_fc1)

        w_out = self._matmul_variable(dim_conv3, 4)
        q_output =  tf.matmul(h_fc1, w_out)

        # q_state_flat = tf.reshape(h_conv1, [-1, self.board_height * self.board_width * self.num_frames])
        # w = self._matmul_variable(self.board_height * self.board_width * self.num_frames, 4)
        # q_output = tf.matmul(q_state_flat, w)

        return q_state, q_output

        # q_state = tf.placeholder(tf.float32, [None, self.board_height, self.board_width, self.num_frames])
        #
        # w_conv1 = self._filter_variable(2, 2, in_channels=self.num_frames, out_channels=1)
        # b_conv1 = self._bias_variable(out_channels=int(w_conv1.get_shape()[-1]))
        # h_conv1 = tf.nn.relu(self._conv2d(q_state, w_conv1, stride=1) + b_conv1)
        # # q_state is [batch_size, height, width, in_channels]
        # # weights is [filter_height, filter_width, in_channels, out_channels]
        # # conv is [batch_size, height/stride, width/stride, out_channels]
        #
        # w_conv2 = self._filter_variable(2, 2, int(w_conv1.get_shape()[-1]), 1)
        # b_conv2 = self._bias_variable(int(w_conv2.get_shape()[-1]))
        # h_conv2 = tf.nn.relu(self._conv2d(h_conv1, w_conv2, stride=1) + b_conv2)
        #
        # #w_conv3 = self._filter_variable(3, 3, 64, 64)
        # #b_conv3 = self._bias_variable(64)
        # #h_conv3 = tf.nn.relu(self._conv2d(h_conv2, w_conv3, stride=1) + b_conv3)
        #
        # #height_after = int(np.ceil(np.ceil(self.board_height / 4.0) / 2.0) / 1.0) # Dimension Size decreases by conv strides
        # #width_after = int(np.ceil(np.ceil(self.board_width / 4.0) / 2.0) / 1.0)   # Dimension Size decreases by conv strides
        # dim_conv2 = int(reduce(mul, h_conv2.get_shape()[1:]))
        # flattened_conv2 = tf.reshape(h_conv2, [-1, dim_conv2])
        #
        # w_fc1 = self._matmul_variable(dim_conv2, np.max([dim_conv2 / 2, 16]))
        # biases4 = self._bias_variable(int(w_fc1.get_shape()[-1]))
        # h_fc1 = tf.nn.relu(tf.matmul(flattened_conv2, w_fc1) + biases4)
        #
        # w_fc2 = self._matmul_variable(int(w_fc1.get_shape()[-1]), 4)
        # b_fc2 = self._bias_variable(int(w_fc2.get_shape()[-1]))
        # h_fc2 = tf.matmul(h_fc1, w_fc2) + b_fc2
        #
        # #action_q_values = tf.placeholder("float", [None, 4]) # 4 == num actions
        # #weights6 = self._matmul_variable(height_after * width_after * 4, 4)
        # #biases6 = self._bias_variable(4)
        # #q_output = tf.matmul(h_fc2, weights6) + biases6
        #
        # return q_state, h_fc2


    @staticmethod
    def _filter_variable(filter_height, filter_width, in_channels, out_channels):
        initial = tf.truncated_normal(shape=[filter_height, filter_width, in_channels, out_channels],
                                      stddev=0.1, mean=0.0)
        #np.ones([filter_height, filter_width, in_channels, out_channels], dtype=np.float32)
        return tf.Variable(initial)

    @staticmethod
    def _matmul_variable(height, width):
        initial = np.zeros([height, width], dtype=np.float32) #tf.truncated_normal(shape=[height, width], stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def _bias_variable(out_channels):
        initial = np.zeros([out_channels], dtype=np.float32) #tf.constant(0.1, shape=[out_channels])
        return tf.Variable(initial)

    @staticmethod
    def _conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

if __name__ == "__main__":
    np.random.seed(1337)
    ds = DeepSnake(3, 3, 2, tf_file_save_path="data/deep_snake/test2.ckpt")
    ds.epsilon = 0.5
    ds.time_penalty = 0.0
    ds.learn_q_function(num_iterations=100, batch_size=50, num_training_steps=10, play_frequency=100)
    ds.save_tf_weights()