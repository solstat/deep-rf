# deep-rf

deep-rf is an open source Python module for Deep Reinforcement Learning built on top of Tensorflow intended for training deep networks to play games.


A typical training session looks like this:

#### Step 1:  Game
```python
import deep_rf as rf

# implement the SinglePlayerGame interface
class MyGame(rf.SinglePlayerGame):
  ...

my_game = MyGame(frame_height=20, frame_width=20)
```

#### Step 2:  Deep network with Tensorflow
```python
# use one of our predefined deep networks
my_q_graph = tf.QGraph.default_q_graph(my_game, num_frames=4)
```

```python
# or define your own using Tensorflow...
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

#### Step 3:  Reward function
```python
#  define function that takes a dictionary of params and returns a numerical value
def my_reward(params):
  ...
  return value
```

Note: `params` should be a dictionary with the following keys: `last_score`, `new_score`, `last_state`, `new_state`, `is_game_over`.


#### Step 4:  Reinforcement learning
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


See the `notebooks` folder for examples.

## Install

You can install our module using
```
pip install git+https://github.com/solstat/snake_learning.git
```

or clone it directly here
```
git clone https://github.com/solstat/snake_learning.git
```

## Dependencies


## About

The project was started in 2016 and was the first project by the Solstat group.  For a complete list of authors, see the `AUTHORS` file.

deep-rf is released under the 


<!-- 

## Todo

* Add some more q graphs

## Ideas

* Soft max in in choosing action (exploitation step)
* Random move iterators that alternate
* Not having the mean in random move iterators grow exponentially

## Tech debt

* Revamp main -> turn into an ipython notebook, hide print function
* Add comments / documentation
* Allow for saving loading (*IMPORTANT!!!*)

-->
