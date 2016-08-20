# deep-rf

deep-rf is an open source Python module for Deep Reinforcement Learning built on top of Tensorflow intended for training deep networks to play games.


#### Step 1:  The game
```python
import deep_rf as rf

# implement the SinglePlayerGame interface
class MyGame(rf.SinglePlayerGame):
  ...

my_game = MyGame(frame_height=20, frame_width=20)
```

#### Step 2:  The deep network
```python
# use one of our predefined deep networks
my_q_graph = tf.QGraph.default_q_graph(my_game, num_frames=4)

# or define your own using Tensorflow...
import tensorflow as tf
g = tf.Graph()
with g.as_default():
  my_input = tf.placeholder(...)
  my_weights = ...
  my_bias = ...
  my_output = tf.nn.relu(tf.matmul(my_input, my_weights) + my_bias)

# ...and passing them into the QGraph constructor
my_q_graph = tf.QGraph(q_input = my_input, q_output = my_output)
```

#### Step 3:  Reward function
```
def my_reward()
```

See the `notebooks` folder for examples.



The project was started in 2016 and was the first project by the Solstat group.  For a complete list of authors, see the `AUTHORS` file.



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
