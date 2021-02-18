import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
from memory import ExperienceMemory




class Agent(object):
    def __init__(self, env, n_actions, input_dims):
        self.env            = env
        self.SCREEN_DIMS    = list(self.env.observation_space.shape)
        self.action_space   = n_actions
        self.gamma          = 0.99
        self.batch_size     = 32
        self.learning_rate  = 0.0000625
        self.FRAMESKIP      = 4
        self.memory= ExperienceMemory(256000, self.FRAMESKIP)
        
        #make network
        self.q_network = self.build_dqn(n_actions, input_dims)
        self.target_network = self.build_dqn(n_actions, input_dims)
        
        #warmup
        self.q_network( tf.ones([1, *input_dims], dtype=tf.float32) )
        self.target_network( tf.ones([1, *input_dims], dtype=tf.float32) )
        
        self.target_network.set_weights( self.q_network.get_weights() )

        #optimizer
        self.optimizer = Adam( lr = self.learning_rate,
                beta_1 = 0.5, beta_2=0.999, epsilon=1e-07, amsgrad=False)

    # This is the same architecture used by deep mind
    def build_dqn(self, n_actions, input_dims):
        inputs = tf.keras.layers.Input( shape=[*input_dims], dtype=tf.float32)
        x = Conv2D( filters=64 , kernel_size=3, strides=1, padding='valid')(inputs)
        x = tf.keras.layers.ReLU()(x) 
        x = Conv2D( filters=128, kernel_size=3, strides=2, padding='valid')(x)
        x = tf.keras.layers.ReLU()(x)
        x = Conv2D( filters=256, kernel_size=3, strides=2, padding='valid')(x)
        x = tf.keras.layers.ReLU()(x)
        x = Conv2D( filters=128, kernel_size=3, strides=2, padding='valid')(x)
        x = tf.keras.layers.ReLU()(x)
        x = Conv2D( filters=64 , kernel_size=3, strides=1, padding='valid')(x)
        x = tf.keras.layers.ReLU()(x)
        x = Flatten()(x)
        
        x = Dense( 512, activation='relu')(x)
        Q = Dense( n_actions, activation=None)(x)
       
        model = tf.keras.Model(inputs=inputs, outputs=Q)
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
        if observation.dtype != np.float32:
            raise ValueError('observation has wrong dtype')

        if self.action_choice_policy(t, eval_flag): 
            action = np.random.randint(0,self.action_space)
        else:
            state = np.array([observation], copy=False, dtype=np.float32)
            actions = self.q_network(state)
            action = np.argmax(actions)
        return action

    def train_step( self, state, action, reward, new_state, done ):
        #Predict both the values we thought we could get. Use the q
        # network for the state and the target network for the next state

        tf.debugging.assert_less_equal( state, tf.constant( 1.0, dtype=tf.float32 ) )
        tf.debugging.assert_less_equal( new_state, tf.constant( 1.0, dtype=tf.float32) )

        action_onehot = tf.one_hot( action, self.action_space, axis=1)
        
        Q_cp = self.q_network( new_state )
        dab  = tf.argmax( Q_cp, axis = 1 )
        dab_onehot = tf.one_hot( dab, self.action_space, axis = 1)
        
        q_t_plus_1 = self.target_network( new_state )
        q_t_plus_1_chosen = tf.multiply( q_t_plus_1, dab_onehot )
        q_t_plus_1_best = tf.reduce_sum( q_t_plus_1_chosen , axis=1 )            
        
        # Dones is 0 or 1, so this acts as a mask so that when the episode
        # is done, we will only take the reward
        # When it is not done, we will take the best value reward of the
        # next state times the future discount
        
        reward = tf.cast( tf.sign( reward ), tf.float32 )
        
        q_target = tf.where(
                tf.cast( done, tf.bool ), 
                reward,
                reward + (self.gamma * q_t_plus_1_best)
        ) 
        
        #q_target = tf.math.sign(reward) + ( 1 - tf.cast(done, tf.float32) ) * ( self.gamma * q_t_plus_1_best )
        
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
        return meanQloss

    def learn(self):
      # First of all, make sure we have enough memories to train on.
        if (self.memory.get_length()-2) > self.batch_size and self.memory.get_length() >= 60000:
            # Get a batch of memories. Each is an array of 32 memories
            state, action, reward, new_state, done = \
                                    self.memory.sample_buffer(self.batch_size)
            return self.train_step( state, action, reward, new_state, done ).numpy()
    
    def reset_env(self):
        o = self.env.reset()
        po = self.process_frame84( o, o)
        curr_state = np.tile(po,[1,1,self.FRAMESKIP])
        curr_state = curr_state.astype(np.float32) / 255.
        return curr_state
    
    def step(self, supstate, t, _eval=False ):
        doneFlag = False
        nextFrame = np.zeros([84,84,1], dtype=np.float32)
        action = self.choose_action( np.expand_dims( supstate, 0), t, _eval )

        i = 0
        tot_reward = 0
        observation = np.zeros([2, *self.SCREEN_DIMS], dtype=np.uint8)

        while (i < self.FRAMESKIP ):
            obs, reward, doneFlag, info = self.env.step( action )
            observation[i%2] = obs
            i+=1
            tot_reward += reward
            newLives = info['ale.lives']

            if doneFlag:
                break
        nextFrame = self.process_frame84( observation[0], observation[1] ).astype(np.float32) / 255.

        supstate[...,:-1] = supstate[...,1:]
        supstate[...,-1 ] = nextFrame[...,0]

        self.memory.store_transition( supstate, action, tot_reward, doneFlag )

        return supstate, tot_reward, doneFlag, info


    def process_frame84(self, obs1, obs2):
        frame = np.maximum( obs1, obs2)
        img   = frame.astype(np.float32)
        img   = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:,:,2] * 0.114

        resized_screen = cv2.resize( img, (84, 84), interpolation=cv2.INTER_LINEAR)
        x_t = np.reshape( resized_screen, [84,84,1])

        return x_t.astype(np.uint8)

    def save_models(self, save_name):
        self.q_network.save(save_name)
    
    # Restore the model and copy parameters to target network
    def load_models(self, save_name):
        self.q_network = tf.keras.models.load_model(save_name)
        self.target_network.set_weights(self.q_network.get_weights())



