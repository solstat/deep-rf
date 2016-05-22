#!/usr/bin/env python
"""
Game Driver

"""

# Import Modules
import numpy as np
from board_state import BoardState, ACTION

wasd_to_action = {
        "w": ACTION.UP,
        "a": ACTION.LEFT,
        "s": ACTION.DOWN,
        "d": ACTION.RIGHT,
        }

# Code To execute
if __name__ == '__main__':
    print "snake.py"
    the_board = BoardState(10,10)
    while not the_board.is_game_over():
        print the_board.get_frame().T
        wasd = raw_input("Move: ")
        if wasd in wasd_to_action.keys():
            the_board.do_action(wasd_to_action[wasd])
        if wasd == "r":
            the_board.reset()
        print "Is game over? %u" % the_board.is_game_over()
        print "Score: %u" % the_board.get_score()
        if wasd == "x":
            break







#EOF
