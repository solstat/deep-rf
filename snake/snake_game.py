"""

Single Player Game class

"""

import numpy as np
from deep_rf import SinglePlayerGame

_SNAKE_ACTION_LIST = ['UP', 'DOWN', 'LEFT', 'RIGHT']

class SnakeGame(SinglePlayerGame):
    def __init__(self, board_height=20, board_width=20):
        SinglePlayerGame.__init__(self, action_list=_SNAKE_ACTION_LIST,
                                  frame_height = board_height,
                                  frame_width = board_width)

        self.initial_location = {'x': np.floor(self.frame_width / 2),
                                 'y': np.floor(self.frame_height / 2)}

        self.snake = Snake(self.initial_location)

        self._new_random_pellet()
        self._score = 0


    @property
    def score(self):
        return self._score


    def do_action(self, action):
        assert(isinstance(action, basestring))
        prev_direction = self.snake.direction
        new_direction = action

        #  Get valid action
        is_180 = new_direction == 'DOWN' and prev_direction == 'UP'
        is_180 = is_180 or new_direction == 'UP' and prev_direction == 'DOWN'
        is_180 = is_180 or new_direction == 'LEFT' and prev_direction == 'RIGHT'
        is_180 = is_180 or new_direction == 'RIGHT' and prev_direction == 'LEFT'
        if is_180:
            new_direction = prev_direction

        #  Do action
        self.snake.direction = new_direction
        self.snake.move()
        if self.is_pellet_scored():
            self.score_pellet()


    def is_pellet_scored(self):
        head = self.snake.body[0]
        if head['x'] == self.pellet['x'] and head['y'] == self.pellet['y']:
            return True
        return False


    def score_pellet(self):
        # self.snake.grow()
        self._new_random_pellet()
        self._score += 1


    def _new_random_pellet(self):
        # Find valid pellet locations
        invalid_pellet_points = self.snake.body
        valid_pellet_points = np.ones((self.frame_width, self.frame_height))
        for invalid_point in invalid_pellet_points:
            valid_pellet_points[int(invalid_point['x']), int(invalid_point['y'])] = 0
        valid_x, valid_y = np.where(valid_pellet_points == 1)

        # Sample one uniformly at random
        num_valid_points = np.sum(valid_pellet_points)
        if num_valid_points == 0:
            raise ValueError("Snake is too big, game should be over")
        new_point_index = np.floor(np.random.rand() * num_valid_points)

        # Update pellet
        self.pellet = {
            'x': valid_x[int(new_point_index)],
            'y': valid_y[int(new_point_index)]
        }


    def get_frame(self):
        frame = np.zeros((self.frame_width, self.frame_height))
        for snake_point in self.snake.body:
            if self._is_in_board(snake_point):
                frame[int(snake_point['x']), int(snake_point['y'])] = 1
        frame[int(self.pellet['x']), int(self.pellet['y'])] = 2
        return frame


    def _is_in_board(self, point):
        if point['x'] < 0:
            return False
        if point['x'] >= self.frame_width:
            return False
        if point['y'] < 0:
            return False
        if point['y'] >= self.frame_height:
            return False
        return True


    def get_score(self):
        return self.score


    def is_game_over(self):
        if self._is_wall_collision():
            return True
        elif self.snake.is_self_collision() and len(self.snake.body) > 2:
            return True
        return False


    def _is_wall_collision(self):
        head = self.snake.body[0]
        if head['y'] < 0 or head['y'] >= self.frame_height:
            return True
        if head['x'] < 0 or head['x'] >= self.frame_width:
            return True
        return False


    def reset(self):
        self.initial_location = {'x': np.floor(self.frame_width / 2),
                                 'y': np.floor(self.frame_height / 2)}
        self.snake = Snake(self.initial_location)
        self._new_random_pellet()
        self._score = 0


    def __str__(self):
        b = self.get_frame().T
        s = "|"
        for i in range(b.shape[0]):
            s += "---"
        s += "|\n"
        for i in reversed(range(b.shape[0])):
            s += "|"
            for j in range(b.shape[1]):
                if b[i, j] == 0:
                    s += "   "
                elif b[i, j] == 1:
                    s += " o "
                else:
                    s += " * "
            s += "|\n"
        s += "|"
        for i in range(b.shape[0]):
            s += "---"
        s += "|"
        return s


class Snake(object):
    """
        Args:
            initial_location (dictionary {'x': int, 'y': int})
        Attributes:
            body (list of dictionaries {'x': int, 'y': int})
            direction (int from action_dict)
        Methods:
            move - move body in direction
            grow - append point to end of body
            is_self_collision - check if snake has collided with itself
    """
    def __init__(self, initial_location):
        self.body = [initial_location]
        self._direction = 'UP'
        self._direction_set = set(_SNAKE_ACTION_LIST)

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        assert(value in self._direction_set)
        self._direction = value

    def move(self):
        head = self.body[0].copy()  # deep copy
        if self._direction == 'UP':
            head['y'] = head['y'] + 1
        elif self._direction == 'DOWN':
            head['y'] = head['y'] - 1
        elif self._direction == 'LEFT':
            head['x'] = head['x'] - 1
        else:
            head['x'] = head['x'] + 1

        self.body.insert(0, head)
        self.body.pop()

    def grow(self):
        tail = self.body[-1]
        self.body.append(tail)

    def is_self_collision(self):
        head = self.body[0]
        if self.body.count(head) > 1:
            return True
        return False
