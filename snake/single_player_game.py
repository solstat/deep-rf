"""

Single Player Game class

"""

import numpy as np


class SinglePlayerGame:
    def __init__(self, action_list, score=0):
        self.action_list = action_list
        self.score = score

    def do_action(self, action):
        raise NotImplementedError('Subclass should define do_action()')

    def get_frame(self):
        raise NotImplementedError('Subclass should define get_frame()')

    def get_score(self):
        raise NotImplementedError('Subclass should define get_score()')

    def is_game_over(self):
        raise NotImplementedError('Subclass should define is_game_over()')

    def reset(self):
        raise NotImplementedError('Subclass should define reset()')


class SnakeGame(SinglePlayerGame):
    def __init__(self, board_height=20, board_width=20):
        SinglePlayerGame.__init__(self, action_list={'UP': 0, 'DOWN': 1,
                                                     'LEFT': 2, 'RIGHT': 3},
                                  score=0)

        self.board_height = board_height
        self.board_width = board_width

        self.initial_location = {'x': np.floor(self.board_width / 2),
                                 'y': np.floor(self.board_height / 2)}

        self.snake = Snake(self.initial_location, self.action_list)

        self._new_random_pellet()


    def do_action(self, action):
        prev_direction = self.snake.direction
        new_direction = action

        #  Get valid action
        is_180 = new_direction == self.action_list['DOWN'] and prev_direction == self.action_list['UP']
        is_180 = is_180 or new_direction == self.action_list['UP'] and prev_direction == self.action_list['DOWN']
        is_180 = is_180 or new_direction == self.action_list['LEFT'] and prev_direction == self.action_list['RIGHT']
        is_180 = is_180 or new_direction == self.action_list['RIGHT'] and prev_direction == self.action_list['LEFT']
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
        self.score += 1

    def _new_random_pellet(self):
        # Find valid pellet locations
        invalid_pellet_points = self.snake.body
        valid_pellet_points = np.ones((self.board_width, self.board_height))
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
        frame = np.zeros((self.board_width, self.board_height))
        for snake_point in self.snake.body:
            if self._is_in_board(snake_point):
                frame[int(snake_point['x']), int(snake_point['y'])] = 1
        frame[int(self.pellet['x']), int(self.pellet['y'])] = 2
        return frame

    def _is_in_board(self, point):
        if point['x'] < 0:
            return False
        if point['x'] >= self.board_width:
            return False
        if point['y'] < 0:
            return False
        if point['y'] >= self.board_height:
            return False
        return True

    def get_score(self):
        return 4

    def is_game_over(self):
        return 5

    def reset(self):
        return 6






class Snake(object):
    """
        Args:
            initial_location (dictionary {'x': int, 'y': int})
            action_list (dictionary {'UP': 0, 'DOWN": 1, ...})
        Attributes:
            body (list of dictionaries {'x': int, 'y': int})
            action_list
            direction (int from action_list)
        Methods:
            move - move body in direction
            grow - append point to end of body
            is_self_collision - check if snake has collided with itself
    """
    def __init__(self, initial_location, action_list):
        self.body = [initial_location]
        self.action_list = action_list
        self.direction = self.action_list['UP']

    def move(self):
        head = self.body[0].copy()  # deep copy
        if self.direction == self.action_list['UP']:
            head['y'] = head['y'] + 1
        elif self.direction == self.action_list['DOWN']:
            head['y'] = head['y'] - 1
        elif self.direction == self.action_list['LEFT']:
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


