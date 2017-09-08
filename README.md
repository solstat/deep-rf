# deep-rf

deep-rf is an open source Python module for deep reinforcement learning with TensorFlow intended to showcase how to train deep neural networks to play games.

This project was started by the Solstat group (see `AUTHORS`) in 2016 as a learning exercise, and is released under the BSD 3-Clause license (see `LICENSE`) intended as a resource for others learning deep reinforcement learning.       


A typical project using this module looks like this:

#### Step 1:  Create a Game
```python
import deep_rf as rf

# extend the SinglePlayerGame class
class MyGame(rf.SinglePlayerGame):
  ...

my_game = MyGame(frame_height=10, frame_width=10)
```

#### Step 2:  Create a deep neural network with TensorFlow
```python
# use a predefined network
params = {
    'filter1': [2, 2], 'out1': 16, 'stride1': 1,
    'filter2': [1, 1], 'out2': 8, 'stride2': 1,
    'filter3': [1, 1], 'out3': 8, 'stride3': 1
}
my_q_graph = rf.QGraph.create_3conv2fc(game=my_game,
                                       num_frames=4,
                                       params=params)
```

```python
# or define your own using TensorFlow
import tensorflow as tf
g = tf.Graph()
with g.as_default():
  my_input = tf.placeholder(...)
  my_weights = ...
  my_bias = ...
  my_output = tf.nn.relu(tf.matmul(my_input, my_weights) + my_bias)

# ...and pass it into the QGraph constructor
my_q_graph = tf.QGraph(q_input=my_input, q_output=my_output)
```

#### Step 3:  Define reward function
```python
#  define reward function that takes these params and returns a float
def my_reward(last_score, new_score, last_state, new_state, is_game_over):
  ...
  return reward
```

#### Step 4:  Train the AI
```python
my_rf = rf.DeepRFLearner(my_game, my_q_graph, my_reward)

while True:
  my_rf.learn_q_function(num_iterations=1000, batch_size=50, num_training_steps=10)
  if raw_input('Continue? (y/n) ') == 'n':
    break
```


See `notebooks` for our example training an AI to play the Snake game.

## Dependencies

deep-rf requires:

- Python 2.7.11
- NumPy 1.11.0
- TensorFlow 0.8.0 

