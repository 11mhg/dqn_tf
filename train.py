import os

os.environ['TF_XLA_FLAGS']="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

import tensorflow as tf
tf.config.set_soft_device_placement(True)

import numpy as np
from agents_2 import Agent
# from agents import Agent
from environment_wrapper import make_env
import sys, shutil
from tqdm import tqdm


num_iter = 0 

ENV_NAME = 'MsPacman-v0'
SAVE_FILE = './' + ENV_NAME + '.h5' 

LOAD_CHECKPOINT = False

evaluation = False

SAVE_FREQUENCY = 2000
MODEL_COPY_FREQUENCY = 2000

BATCH_SIZE = 32 
 
env = make_env( ENV_NAME )

agent = Agent( n_actions=env.action_space.n, input_dims=(4,84,84) )

if LOAD_CHECKPOINT:
    agent.load_models( SAVE_FILE )

agent.save_models( SAVE_FILE ) 


scores = []
losses = []
pbar = tqdm( total = int(5e6) )
pbar.update( num_iter )
best_score = -np.inf
while True:

    if num_iter % MODEL_COPY_FREQUENCY == 0:
        agent.target_network.set_weights( agent.q_network.get_weights() )
    
    done = False
    observation = env.reset()
    score = 0

    while not done:
        pbar.update(1)
        num_iter += 1
        action = agent.choose_action( observation, num_iter )
        observation_, reward, done, info = env.step( action )
        score += reward
        agent.store_transition(
            observation, action, 
            reward, int(done)
        )
        l = agent.learn()
        if l is not None:
            losses.append( l )
        observation = observation_
    
    scores.append(score)
    avg_score = np.mean( scores[-50:] )
    avg_losses = np.mean( losses[-50:] )
    
    if avg_score > best_score:
        best_score = avg_score
        agent.save_models( SAVE_FILE ) 
    pbar.set_description(
        'l: %.4f score: %d average score: %.3f, eps %.2f' % (avg_losses, score, avg_score, agent.get_eps(num_iter) )
    )

    if num_iter > 5e6:
        break






