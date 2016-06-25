#!/usr/bin/env python
"""
Provides Board State and Snake Class

"""

# Import Modules
import numpy as np
from collections import namedtuple
from enum import Enum

# Directions Struct
class ACTION(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

ACTION_LIST = [ACTION.UP, ACTION.DOWN, ACTION.LEFT, ACTION.RIGHT]

_UP = -1
_DOWN = 1
_LEFT = -1
_RIGHT = 1

# Point Class
Point = namedtuple('Point', ['x', 'y'])

class BoardState(object):
    """
    Args:
      board_height (int)
      board_width (int)
    Attributes:
      snake (Snake object)
      pellet (Point)
      score (int)

    Methods:
      do_action(ACTION) - apply input action to snake
      get_frame() - return frame
      get_score() - return score
      is_game_over() - check if game is over
      get_direction() - return direction from snake
      reset() - reset game
    """
    def __init__(self, board_height=20, board_width=20):
        self.board_height = board_height
        self.board_width = board_width
        self.score = 0
        initial_location = Point(
                x = np.floor(board_width/2),
                y = np.floor(board_height/2))
        self.snake = Snake(initial_location = initial_location)
        self._new_random_pellet()
        return

    def do_action(self, action):
        # Prevent snake from 180-turn around
        direction = self._get_valid_action(action)
        self.snake.set_direction(new_direction = direction)
        self.snake.move()
        if self.is_pellet_scored():
            self.score_pellet()
        return

    def _get_valid_action(self, action):
        prev_direction = self.get_direction()
        if action == ACTION.DOWN and prev_direction == ACTION.UP:
            return ACTION.UP
        elif action == ACTION.UP and prev_direction == ACTION.DOWN:
            return ACTION.DOWN
        elif action == ACTION.LEFT and prev_direction == ACTION.RIGHT:
            return ACTION.RIGHT
        elif action == ACTION.RIGHT and prev_direction == ACTION.LEFT:
            return ACTION.LEFT
        else:
            return action

    def is_pellet_scored(self):
        head = self.snake.get_head()
        if head.x == self.pellet.x and head.y == self.pellet.y:
            return True
        return False

    def score_pellet(self):
        self.snake.grow()
        self._new_random_pellet()
        self.score += 1
        return

    def _new_random_pellet(self):
        """ Random location for pellet, not in snake """
        # Find valid pellet locations
        invalid_pellet_points = self.snake.get_body()
        valid_pellet_points = np.ones((self.board_width, self.board_height))
        for invalid_point in invalid_pellet_points:
            valid_pellet_points[int(invalid_point.x), int(invalid_point.y)] = 0
        valid_x, valid_y = np.where(valid_pellet_points == 1)

        # Sample one uniformly at random
        num_valid_points = np.sum(valid_pellet_points)
        if num_valid_points == 0:
            raise ValueError("Snake is too big, game should be over")
        new_point_index = np.floor(np.random.rand()*num_valid_points)

        # Update pellet
        self.pellet = Point(
                x = valid_x[int(new_point_index)],
                y = valid_y[int(new_point_index)])
        return

    def is_game_over(self):
        if self._wall_collision():
            return True
        if self.snake.self_collision() and len(self.snake.get_body()) > 2:
            return True
        return False

    def _wall_collision(self):
        head = self.snake.get_head()
        if head.x < 0 or head.x >= self.board_width:
            return True
        if head.y < 0 or head.y >= self.board_height:
            return True
        return False

    def get_frame(self):
        frame = np.zeros((self.board_width, self.board_height))
        for snake_point in self.snake.get_body():
            if self._is_in_board(snake_point):
                frame[int(snake_point.x), int(snake_point.y)] = 1
        frame[int(self.pellet.x), int(self.pellet.y)] = 2
        return frame

    def _is_in_board(self, point):
        if point.x < 0:
            return False
        if point.x >= self.board_width:
            return False
        if point.y < 0:
            return False
        if point.y >= self.board_height:
            return False
        return True

    def get_score(self):
        return self.score

    def get_direction(self):
        return self.snake.get_direction()

    def reset(self):
        initial_location = Point(
                x = np.floor(self.board_width/2),
                y = np.floor(self.board_height/2))
        self.snake = Snake(initial_location = initial_location)
        self._new_random_pellet()
        self.score = 0
        return

class Snake(object):
    """
    Args:
      initial_location (Point)
    Attributes:
      body (list of Points)
      direction (ACTION)
    Methods:
      get_head - return the head of snake
      get_body - return the body of snake
      get_direction - return direction of snake
      self_collision - check if snake has collided with itself
      grow - append point to end of body
      move - move body in direction
    """
    def __init__(self, initial_location):
        self.body = [initial_location]
        self.direction = ACTION.UP
        return

    def get_head(self):
        return self.body[0]

    def get_body(self):
        return self.body

    def get_direction(self):
        return self.direction

    def set_direction(self, new_direction):
        """ new_direction is an ACTION """
        self.direction = new_direction
        return

    def self_collision(self):
        head = self.body[0]
        if self.body.count(head) > 1:
            return True
        return False

    def grow(self):
        self.body.append(self.body[-1])
        return

    def move(self):
        new_head = self.body[0]
        if self.direction == ACTION.UP:
            new_head = new_head._replace(y = new_head.y + _UP)
        elif self.direction == ACTION.DOWN:
            new_head = new_head._replace(y = new_head.y + _DOWN)
        elif self.direction == ACTION.LEFT:
            new_head = new_head._replace(x = new_head.x + _LEFT)
        elif self.direction == ACTION.RIGHT:
            new_head = new_head._replace(x = new_head.x + _RIGHT)
        else:
            raise ValueError("Unrecognized direction")

        self.body.insert(0, new_head)
        self.body.pop()
        return


# Code To execute
if __name__ == '__main__':
    print "snake.py"





#EOF
