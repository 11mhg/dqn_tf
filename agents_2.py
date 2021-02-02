import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
from memory import ExperienceMemory




class Agent(object):
    def __init__(self, n_actions, input_dims):
        self.action_space = n_actions
        self.gamma = 0.99
        self.batch_size = 32
        self.learning_rate = 0.0000625
        self.memory= ExperienceMemory(64000)
        self.q_network = self.build_dqn(n_actions, input_dims)
        self.target_network = self.build_dqn(n_actions, input_dims)
        self.optimizer = Adam( lr = self.learning_rate )

    # This is the same architecture used by deep mind
    def build_dqn(self, n_actions, input_dims):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation='relu',
                        input_shape=(*input_dims,), data_format='channels_first'))
        model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation='relu',
                        data_format='channels_first'))
        model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu',
                        data_format='channels_first'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(n_actions))
        return model

    def store_transition(self, state, action, reward, done):
        self.memory.store_transition(state, action, reward, done)
    
    def get_eps(self, t=0):
        t = max(0, t - 64000)
        return max( 0.05, 1.0 - (t * (1-0.05)/1000000.) )

    def action_choice_policy(self,t = 0, eval_flag=False):
        if eval_flag:
            return False
        if t < 64000:
            return True
        eps = self.get_eps( t )
        if np.random.random() < eps:
            return True
        return False

    # The agent will make use of our ExplorerExploiter to choose either
    # Random actions or the action that gives the highest Q value
    def choose_action(self, observation,t = 0, eval_flag=False):
        if self.action_choice_policy(t, eval_flag): 
            action = np.random.randint(0,self.action_space)
        else:
            state = np.array([observation], copy=False, dtype=np.float32)
            actions = self.q_network(state)
            action = np.argmax(actions)
        return action
    
    def learn(self):
      # First of all, make sure we have enough memories to train on.
        if self.memory.get_length() > self.batch_size:
            # Get a batch of memories. Each is an array of 32 memories
            state, action, reward, new_state, done = \
                                    self.memory.sample_buffer(self.batch_size)

            #Predict both the values we thought we could get. Use the q
            # network for the state and the target network for the next state
            action_onehot = tf.one_hot( action, self.action_space, axis=1)

            q_t = self.q_network( state )
            q_t_selected = tf.reduce_sum( q_t * action_onehot, axis = 1)

            q_t_plus_1 = self.target_network( new_state )
            q_t_plus_1_best = tf.reduce_max( q_t_plus_1, axis=1 )            

            # Dones is 0 or 1, so this acts as a mask so that when the episode
            # is done, we will only take the reward
            # When it is not done, we will take the best value reward of the
            # next state times the future discount

            reward = tf.cast( tf.sign( reward ), tf.float32 )

            q_target = (1.0 - done ) * q_t_plus_1_best
            q_target = reward + (self.gamma * q_target)



            # finally, train the network to backpropogate the loss
            with tf.GradientTape() as tape:
                q_var = self.q_network(state)
                qvar_s_a1 = tf.reduce_sum( tf.multiply( q_var, action_onehot ), axis=1)

                error_q = q_target - qvar_s_a1
                Qhuber_bool1 = tf.less(tf.abs(error_q), 1.0)
                Qhuber_loss = tf.where( Qhuber_bool1, 0.5 * tf.square(error_q), tf.abs( error_q ) - 0.5 )
                meanQloss = tf.reduce_sum( Qhuber_loss)

            Q_grad = tape.gradient( meanQloss, self.q_network.trainable_weights )
            self.optimizer.apply_gradients( zip(Q_grad, self.q_network.trainable_weights) )
            return meanQloss.numpy() 


    def save_models(self, save_name):
        self.q_network.save(save_name)
    
    # Restore the model and copy parameters to target network
    def load_models(self, save_name):
        self.q_network = tf.keras.models.load_model(save_name)
        self.target_network.set_weights(self.q_network.get_weights())
