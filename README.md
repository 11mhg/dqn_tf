# DQN Tensorflow 2

This is a tensorflow 2 dqn implementation based off the DQN paper. This uses the functional keras api to build the model. 

Agents.py uses keras train\_on\_batch which, for reasons I'm not aware of, takes 2.5x longer to train than Agents\_2.py.

Agents\_2.py uses a custom training loop and tf.GradientTape() to modify the model weights.





Author: Mohammed Hamada Gasmallah
