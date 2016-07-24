#!/usr/bin/env python
"""
Game Driver

"""

# Import Modules
from board_state import BoardState, ACTION
import sys

class _Getch:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


wasd_to_action = {
        "w": ACTION.UP,
        "a": ACTION.LEFT,
        "s": ACTION.DOWN,
        "d": ACTION.RIGHT,
        }

def board_to_string(b):
    s = "|"
    for i in range(b.shape[0]):
        s += "---"
    s += "|\n"
    for i in range(b.shape[0]):
        s += "|"
        for j in range(b.shape[1]):
            if b[i,j] == 0:
                s += "   "
            elif b[i,j] == 1:
                s += " o "
            else:
                s += " * "
        s += "|\n"
    s += "|"
    for i in range(b.shape[0]):
        s += "---"
    s += "|"
    return s
# Code To execute
if __name__ == '__main__':
    print "snake.py"
    the_board = BoardState(10,10)
    getch = _Getch()
    while not the_board.is_game_over():
        sys.stdout.write("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        sys.stdout.write("Score: " + str(the_board.get_score()) + "\n")
        sys.stdout.write(board_to_string(the_board.get_frame().T))
        sys.stdout.flush()

        wasd = getch()

        if wasd == "x":
            break
        elif wasd == "r":
            the_board.reset()
            continue

        the_board.do_action(wasd_to_action[wasd])



#EOF
