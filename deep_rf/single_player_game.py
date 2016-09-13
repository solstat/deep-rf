"""

Single Player Game class

"""

class SinglePlayerGame:
    """ A virtual class for single player games

    This contains the class skeleton for a single player game.
    To be used in deep reinforcement learning.

    Args:
        action_list (list): set of game actions
        frame_width (int): non-negative size of game window width
        frame_height (int): non-negative size of game window height

    Attributes:
        action_list (list): set of game actions
        action_dict (dict): enumeration of action_list

    See Also:
        deep_rf.deep_rf_learner.DeepRFLearner
    """

    def __init__(self, action_list, frame_height, frame_width):
        self.action_list = action_list
        self.action_dict = {self.action_list[i]: i for i in
                            range(len(self.action_list))}
        self._frame_height = frame_height
        self._frame_width = frame_width

    @property
    def frame_height(self):
        """ frame_height (int): non-negative size of game window height """
        return self._frame_height

    @property
    def frame_width(self):
        """ frame_width (int): non-negative size of game window width """
        return self._frame_width

    @property
    def score(self):
        """ score (double): current score of game """
        raise NotImplementedError('Subclass should define get_score()')

    def do_action(self, action):
        """ Apply player's selected action to current game.

        Args:
            action (Action): action to perform

        Returns:
            None: applies action to game
        """
        raise NotImplementedError('Subclass should define do_action()')

    def get_frame(self):
        """ Return the pixels for the current game

        Returns:
            frame (ndarray): returns the frame_height by frame_width game window.
        """
        raise NotImplementedError('Subclass should define get_frame()')

    def is_game_over(self):
        """ Return whether the game has ended

        Returns:
            isGameOver (bool): returns whether the game has ended
        """
        raise NotImplementedError('Subclass should define is_game_over()')

    def reset(self):
        """ Start a new game. """
        raise NotImplementedError('Subclass should define reset()')

