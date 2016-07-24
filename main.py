"""

Test

"""

import numpy as np
import snake.snake_game as snake
from snake.ascii_snake import board_to_string
import deep_rf as rf

frame_height = 3
frame_width = 3
num_frames = 2

my_game = snake.SnakeGame(board_height=frame_height, board_width=frame_width)
my_q_graph = rf.QGraph.default_q_graph(my_game, num_frames=num_frames)

def my_reward(params):
    return params['new_score'] - params['last_score'] + \
           (0.0 if params['is_game_over'] else -1.0)

my_rf = rf.DeepRFLearner(my_game, my_q_graph, my_reward)


def play_one_game(deep_rf_learner):

    game = snake.SnakeGame(frame_height, frame_width)
    first_frame = game.get_frame()
    state_padding = [np.zeros(first_frame.shape) for _ in
                     range(num_frames - 1)]
    current_state = rf.State(frames_tuple=tuple(state_padding) + (first_frame,))

    while True:

        print board_to_string(game.get_frame().T)

        action = deep_rf_learner.predict(X=current_state, type='action')

        r = raw_input("Press q to quit or Press r to reset: ")
        if r == 'q':
            break
        elif r == 'r' or game.is_game_over():
            return play_one_game(deep_rf_learner)
        else:
            game.do_action(action)
            current_state = current_state.new_state_from_old(
                game.get_frame())
            q_values = deep_rf_learner.predict(X=current_state)
            action = deep_rf_learner.predict(X=current_state, type='action')
            print "\n\n\n\n\n\n\n"
            print "Current Q Values: " + str(q_values)
            print game.action_list
            print "Next Action:" + action
    return



while True:
    my_rf.learn_q_function(num_iterations=500,
                           batch_size=50,
                           num_training_steps=10,
                           epsilon_multiplier=1.0)
    play_one_game(my_rf)

