# deep-rf

deep-rf is an open source Python module for deep reinforcement learning with TensorFlow intended to showcase how to train deep neural networks to play games.  This project was started by the Solstat group (see `AUTHORS`) in 2016 as a learning exercise, and is released under the 3-Clause BSD license (see `LICENSE`) intended as (hopefully) educational material for others learning deep reinforcement learning.       


A typical project using this module looks like this:

#### Step 1:  Create a Game
```python
import deep_rf as rf

# extend the SinglePlayerGame class
class MyGame(rf.SinglePlayerGame):
  ...

my_game = MyGame(frame_height=20, frame_width=20)
```

#### Step 2:  Create a deep neural network with TensorFlow
```python
# use one of our predefined deep networks
my_q_graph = rf.QGraph.default_q_graph(my_game, num_frames=4)
```

```python
# or define your own using TensorFlow...
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
#  define function that takes a dictionary of params and returns a numerical value
def my_reward(params):
  ...
  return value
```

Note: `params` should be a dictionary with the following keys: `last_score`, `new_score`, `last_state`, `new_state`, `is_game_over`.


#### Step 4:  Train the AI
```python
my_rf = rf.DeepRFLearner(my_game, my_q_graph, my_reward)

while True:
  my_rf.learn_q_function(num_iterations=1000, batch_size=50, num_training_steps=10)
  if raw_input('Continue? (y/n) ') == 'n':
    break
```

#### Step 5:  Save results

```python
my_rf.save_tf_weights(file_path=MY_FILE_PATH)
```

See `notebooks` for our example training an AI to play the Snake game.

## Dependencies

deep-rf requires:

- Python 2.7.11
- NumPy 1.11.0
- TensorFlow 0.8.0 

