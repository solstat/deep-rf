
#!/usr/bin/env python
"""
Does Deep Q-Learning for Snake

"""

# Import Modules
import numpy as np
import tensorflow as tf
from operator import mul
import os
from single_player_game import SinglePlayerGame
from q_graph import QGraph

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

class DeepRFLearner(object):

    def __init__(self, game, q_graph, reward_function, file_save_path=None):
        """
        Args:
            game:
            q_graph:
            num_frames:
            reward_function: A function taking a dictionary of parameters and returning a double.
                             Dictionary args include 'last_score', 'new_score', 'last_state', 'new_state',
                              'is_game_over'.
            file_save_path:
        """
        self.epsilon = 0.8
        self.gamma = 0.9
        self.file_save_path = file_save_path

        #  TODO: game.copy()
        assert(isinstance(game, SinglePlayerGame))
        assert (isinstance(q_graph, QGraph))
        self._game = game
        self._q_graph = q_graph

        self._num_frames = int(self._q_graph.q_input.get_shape()[-1])

        self._reward_function = reward_function

        self._experience_replay = []

        with self._q_graph.graph.as_default() as g:
            with g.name_scope("non_q_graph_ops"):
                self._y_obs, self._action_indices, self._loss = self._init_training_loss_operation()
                self._optimizer = tf.train.AdamOptimizer().minimize(self._loss)
                self._saver = tf.train.Saver(var_list=self._q_graph.var_list)
                self._sess = tf.Session(graph=g)

                # TODO: DeepRF.start_session() and only load correct variables
                if file_save_path is not None and os.path.exists(file_save_path):
                    self._saver.restore(self._sess, file_save_path)
                    print("Model restored from " + file_save_path + ".")
                else:
                    self._sess.run(tf.initialize_all_variables())


    def __del__(self):
        self._sess.close()


    def save_tf_weights(self):
        if self.file_save_path is not None:
            self._saver.save(self._sess, self.file_save_path)


    def _init_training_loss_operation(self):
        y_obs = tf.placeholder(dtype=tf.float32, shape=[None])
        action_indices = tf.placeholder(dtype=tf.int64, shape=[None])
        action_one_hot = tf.one_hot(action_indices, 4, on_value=1.0, off_value=0.0)
        y_pred = tf.reduce_sum(self._q_graph.q_output * action_one_hot, reduction_indices=1)
        loss = tf.reduce_mean((y_obs - y_pred)**2)
        return y_obs, action_indices, loss


    def learn_q_function(self, num_iterations=1000, batch_size=50,
                         num_training_steps=10, epsilon_multiplier=1.0):
        # For Training Time
            # Get next sample -> List of ExperienceTuples
            # Get a minibatch -> partially Optimize Q for loss

        # Return Q or Q parameters

        experience_tuple_generator = self.get_next_experience_tuple()
        for it in xrange(num_iterations):
            print it
            for __ in xrange(batch_size):
                self._experience_replay.append(experience_tuple_generator.next())
            experience_batch = np.random.choice(self._experience_replay, batch_size, replace=False)
            actions_batch = [self._game.action_dict[et.action] for et in experience_batch]
            states_batch = [et.state.to_array() for et in experience_batch]
            y_targets = self._get_target_values(experience_batch)

            for __ in xrange(num_training_steps):
                self._sess.run(self._optimizer, feed_dict={self._q_graph.q_input: states_batch,
                                                             self._action_indices: actions_batch,
                                                             self._y_obs: y_targets})

            self.epsilon *= epsilon_multiplier

        return

    def _get_target_values(self, experience_batch):
        """
        Args:
            experience_batch:  list of ExperienceTuples

        Returns:
            y_target:   np.ndarray of [batch_size, r + max Q(s')]
        """
        rewards = np.array([et.reward for et in experience_batch])
        states = [
            et.next_state.to_array() if et.next_state is not None else et.state.to_array()
            for et in experience_batch]
        q_values = self._sess.run(self._q_graph.q_output,
                                  feed_dict={self._q_graph.q_input: states})
        game_not_over_indicator = np.array(
            [1.0 if et.next_state is not None else 0.0 for et in
             experience_batch])
        y_target = rewards + self.gamma * np.max(q_values,
                                                 axis=1) * game_not_over_indicator
        return y_target

    def get_next_experience_tuple(self):
        """ Yield the Experience Tuple for training Q
        yields:
          experience_tuple (Experience Tuple) - current state, action, reward, new_state
        """
        while True:
            self._game.reset()
            first_frame = self._game.get_frame()
            state_padding = [np.zeros(first_frame.shape) for _ in range(self._num_frames - 1)]
            current_state = State(tuple(state_padding) + (first_frame,))

            while not self._game.is_game_over():
                action = self._get_action_with_noise(current_state)
                last_score = self._game.score
                self._game.do_action(action)
                new_state = current_state.new_state_from_old(self._game.get_frame())
                new_score = self._game.score
                reward = self._reward_function({"last_score":last_score,
                                                "new_score":new_score,
                                               "last_state":current_state,
                                                "new_state":new_state,
                                               "is_game_over":self._game.is_game_over()})
                if self._game.is_game_over():
                    yield ExperienceTuple(current_state, action, reward, None)
                else:
                    yield ExperienceTuple(current_state, action, reward, new_state)
                current_state = new_state


    def _get_action_with_noise(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self._game.action_list)
        else:
            return self.predict(X=state, type='action')

    def predict(self, X, type='q_value'):
        if isinstance(X, State):
            q_state = np.array([X.to_array()])
        elif isinstance(X, list):
            q_state = np.array([state.to_array() for state in X])
        else:
            raise TypeError

        q_values = self._sess.run(self._q_graph.q_output,
                                  feed_dict={self._q_graph.q_input: q_state})
        if type == "q_value":
            return_values = q_values
        elif type == 'action':
            return_values = [
                self._game.action_list[np.argmax(q_values[i, :])]
                for i in xrange(q_values.shape[0])
            ]
        else:
            raise NotImplementedError

        if isinstance(X, State):
            return return_values[0]
        else:
            return return_values


