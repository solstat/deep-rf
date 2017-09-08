#
# Author: Christopher Aicher <aicherc@uw.edu>
# License: BSD 3 clause
#

import numpy as np
import snake.snake_game as snake
import deep_rf as rf
from gif_maker import GifMaker
import time

# snake rf parameters
frame_height = 4
frame_width = 4
num_frames = 2

# gif parameters
num_gif_frames = 50


# Define Deep RF
my_game = snake.SnakeGame(board_height=frame_height, board_width=frame_width)
my_q_graph = rf.QGraph(name='snake_default', frame_height=frame_height,
                       frame_width=frame_width, num_frames=num_frames,
                       num_actions=len(my_game.action_list))
def my_reward(params):
    return params['new_score'] - params['last_score'] + \
           (-1.0 if params['is_game_over'] else 0.0) - .001
my_rf = rf.DeepRFLearner(my_game, my_q_graph, my_reward)

def produce_gif(deep_rf_learner, filename, num_gif_frames=num_gif_frames):
    game = snake.SnakeGame(frame_height, frame_width)
    current_state = get_init_state(game)
    my_gif_maker = GifMaker(game=game)

    action = print_frame_and_get_action(current_state, game, deep_rf_learner)
    my_gif_maker.add_frame()
    for _ in xrange(0, num_gif_frames):
        time.sleep(0.25)
        if game.is_game_over():
            game.reset()
            current_state = get_init_state(game)
            continue
        else:
            game.do_action(action)
            current_state = current_state.new_state_from_old(game.get_frame())
            action = print_frame_and_get_action(current_state, game,
                    deep_rf_learner)
            my_gif_maker.add_frame()

    my_gif_maker.save_gif(filename)
    return

# Helper functions
def get_init_state(game):
    first_frame = game.get_frame()
    state_padding = [np.zeros(first_frame.shape) for _ in
                     range(num_frames - 1)]
    init_state = rf.State(
        frames_tuple=tuple(state_padding) + (first_frame,))
    return init_state

def print_frame_and_get_action(state, game, deep_rf_learner):
    print "\n" * 20 + str(game)
    print "\nQ-val with actions: " + \
          str(dict(zip(game.action_list,
                       np.round(deep_rf_learner.evaluate_q_function(
                           state=state), 3))))
    action = deep_rf_learner.choose_action(state=state)
    print "Next Action: " + action
    return action


if __name__ is "__main__":
    print "Train for 10 iterations"
    my_rf.learn_q_function(num_iterations=10,
                       batch_size=100,
                       num_training_steps=100)
    produce_gif(my_rf, "gif_maker/10Iter.gif")

    print "Train for 100 iterations"
    my_rf.learn_q_function(num_iterations=90,
                       batch_size=100,
                       num_training_steps=100)
    produce_gif(my_rf, "gif_maker/100Iter.gif")

    print "Train for 400 iterations"
    my_rf.learn_q_function(num_iterations=300,
                       batch_size=100,
                       num_training_steps=100)
    produce_gif(my_rf, "gif_maker/400Iter.gif")

