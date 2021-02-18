import os

os.environ['TF_XLA_FLAGS']="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

import tensorflow as tf
tf.config.set_soft_device_placement(True)

import numpy as np
from agents import Agent
from environment_wrapper import make_env
import sys, shutil
from tqdm import tqdm


num_iter = 0 

ENV_NAME = 'MsPacman-v0'
SAVE_FILE = './' + ENV_NAME + '.h5' 

LOAD_CHECKPOINT = True

evaluation = False

SAVE_FREQUENCY = 100000
MODEL_COPY_FREQUENCY = 2500

BATCH_SIZE = 32 
 
env = make_env( ENV_NAME )

agent = Agent( n_actions=env.action_space.n, input_dims=(4,84,84) )

if LOAD_CHECKPOINT:
    try:
        agent.load_models( SAVE_FILE )
    except Exception:
        print("Model not found...")
agent.save_models( SAVE_FILE ) 

writer = tf.summary.create_file_writer('./logs/train')

scores = []
losses = []
pbar = tqdm( total = int(5e6) )
pbar.update( num_iter )
best_score = -np.inf

num_episodes = 0
while True:
    num_episodes+=1 
    done = False
    observation = env.reset()
    score = 0
   
    
    train_eval = False

    while not done:
        pbar.update(1)
        num_iter += 1
        action = agent.choose_action( observation, num_iter, eval_flag=train_eval )
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
        
        if num_iter % MODEL_COPY_FREQUENCY == 0:
            agent.target_network.set_weights( agent.q_network.get_weights() )

    scores.append(score)
    avg_score = np.mean( scores[-100:] )
    avg_losses = np.mean( losses[-100:] )

    if avg_score > best_score or num_iter % SAVE_FREQUENCY == 0:
        best_score = avg_score
        agent.save_models( SAVE_FILE )

    if len(losses) > 0:
        with writer.as_default():
            tf.summary.scalar("loss", avg_losses, step=num_iter)
            tf.summary.scalar("score", avg_score, step=num_iter)
            tf.summary.scalar('eps', agent.get_eps(num_iter), step=num_iter)

        pbar.set_description(
            '%d - l: %.4f score: %d average score: %.3f, eps %.2f eval? %s' % (num_episodes,
                    avg_losses, score, avg_score, agent.get_eps(num_iter), str(train_eval) )
        )


    if num_iter > 5e6:
        break






