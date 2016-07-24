"""

Single Player Game class

"""



class SinglePlayerGame:
    def __init__(self, action_list, frame_height, frame_width):
        self.action_list = action_list
        self.action_dict = {self.action_list[i]: i for i in
                            range(len(self.action_list))}
        self._frame_height = frame_height
        self._frame_width = frame_width

    @property
    def frame_height(self):
        return self._frame_height

    @property
    def frame_width(self):
        return self._frame_width

    @property
    def score(self):
        raise NotImplementedError('Subclass should define get_score()')

    def do_action(self, action):
        raise NotImplementedError('Subclass should define do_action()')

    def get_frame(self):
        raise NotImplementedError('Subclass should define get_frame()')

    def is_game_over(self):
        raise NotImplementedError('Subclass should define is_game_over()')

    def reset(self):
        raise NotImplementedError('Subclass should define reset()')

