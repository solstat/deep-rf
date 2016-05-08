#!/usr/bin/env python
"""
Board State

"""

# Import Modules
import numpy as np
from collections import namedtuple

_NORTH_DIR = 0
_EAST_DIR = 1
_SOUTH_DIR = 2
_WEST_DIR = 3
_DO_NOTHING = -1

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
      do_action(int) - apply input action to snake
      check_score_pellet() - check if snake eats pellet
      score_pellet() - update board when pellet is eaten
      _new_random_pellet() - generate new pellet
      check_game_over() - check if game is over
      _wall_collision() - check if snake has collided with wall
      get_frame() - return frame
      get_score() - return score
      reset() - reset game
    """
    def __init__(self, board_height, board_width):
        raise NotImplementedError("not yet implemented")
        return

class Snake(object):
    """
    Args:
      initial_location (Point)
    Attributes:
      body (list of Points)
      direction (int)
    Methods:
      get_head - return the head of snake
      get_body - return the body of snake
      self_collision - check if snake has collided with itself
      grow - append point to end of body
      move - move body in direction
    """
    def __init__(self, initial_location):
        raise NotImplementedError("not yet implemented")
        return

# Code To execute
if __name__ == '__main__':
    print "snake.py"





#EOF
