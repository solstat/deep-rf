"""

Test

"""

import numpy as np
import snake.snake_game as snake
import deep_rf as rf

frame_height = 4
frame_width = 4
num_frames = 2

my_game = snake.SnakeGame(board_height=frame_height, board_width=frame_width)
my_q_graph = rf.QGraph.default_q_graph(my_game, num_frames=num_frames)

def my_reward(params):
    return params['new_score'] - params['last_score'] + \
           (-1.0 if params['is_game_over'] else 0.0) - .001

my_rf = rf.DeepRFLearner(my_game, my_q_graph, my_reward)


def play_one_game(deep_rf_learner):

    game = snake.SnakeGame(frame_height, frame_width)
    first_frame = game.get_frame()
    state_padding = [np.zeros(first_frame.shape) for _ in
                     range(num_frames - 1)]
    current_state = rf.State(frames_tuple=tuple(state_padding) + (first_frame,))

    def print_frame_and_get_action(state):
        print "\n" * 20 + str(game)
        print "\nQ-val with actions: " + \
            str(dict(zip(game.action_list,
                np.round(deep_rf_learner.evaluate_q_function(state=state), 3))))
        action = deep_rf_learner.choose_action(state=state)
        print "Next Action: " + action
        return action

    action = print_frame_and_get_action(current_state)

    while True:
        r = raw_input("Press q to quit or Press r to reset: ")
        if r == 'q':
            break
        elif r == 'r' or game.is_game_over():
            return play_one_game(deep_rf_learner)
        else:
            game.do_action(action)
            current_state = current_state.new_state_from_old(game.get_frame())
            action = print_frame_and_get_action(current_state)
    return


while True:
    my_rf.learn_q_function(num_iterations=50,
                           batch_size=1000,
                           num_training_steps=100)
    play_one_game(my_rf)

