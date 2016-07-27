"""
Game Driver
"""

from snake.snake_game import SnakeGame
import sys, tty, termios, select, time
import curses

def get_input(timeout):
  fd = sys.stdin.fileno()
  old_settings = termios.tcgetattr(fd)
  try:
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
      ch = sys.stdin.read(1)
    else:
      ch = None
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
  return ch

wasd_to_action = { "w": "UP", "a": "LEFT",
                   "s": "DOWN", "d": "RIGHT" }

# Code To execute
if __name__ == '__main__':
    width = 10 if len(sys.argv) == 1 else int(sys.argv[1])
    height = 10 if len(sys.argv) < 3 else int(sys.argv[2])

    game = SnakeGame(width, height)
    last_action = "w"
    try:
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()

        frame_height = len(str(game).split("\n"))
        a = time.time()
        while time.time() - a < 3:
            stdscr.addstr(6, 0, str(game))
            stdscr.addstr(6 + frame_height, 0, "Score: " + str(game.score) + "\n")
            stdscr.addstr(7 + frame_height, 0,
                          "Starting in %.1f" % (3.0 - (time.time() - a)) +
                          " seconds.")
            stdscr.refresh()

        while True:
            stdscr.addstr(6, 0, str(game))
            stdscr.addstr(6 + frame_height, 0, "Score: " + str(game.score) + " " * 10)
            stdscr.addstr(7 + frame_height, 0, "Go!" + " " * 30)
            stdscr.refresh()

            wasd = get_input(1.0 / 5.0)

            if wasd == "x":
                break
            elif wasd == "r":
                game.reset()
                continue
            elif wasd is None or not wasd_to_action.has_key(wasd):
                wasd = last_action

            last_action = wasd
            game.do_action(wasd_to_action[wasd])

            if game.is_game_over():
                stdscr.addstr(6, 0, str(game))
                stdscr.addstr(6 + frame_height, 0, "Score: " + str(game.score))
                stdscr.addstr(7 + frame_height, 0,
                              "Game over! Play another? (y/n)")
                stdscr.refresh()

                play_again = get_input(60)
                while play_again.lower() != 'y' and play_again.lower() != 'n':
                    play_again = get_input(60)

                if play_again.lower() == "y":
                    game.reset()
                    last_action = "w"
                else:
                    break
    finally:
        curses.echo()
        curses.nocbreak()
        curses.endwin()

    print str(game)
    print "Score: " + str(game.score)
    print "Game over! Play another? (y/n)"