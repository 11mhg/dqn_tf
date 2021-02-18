import cv2
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


ENV_NAME = 'MsPacman-v0'
SAVE_FILE = './' + ENV_NAME + '.h5' 

LOAD_CHECKPOINT = True#False

evaluation = True 

env = make_env( ENV_NAME )

agent = Agent( n_actions=env.action_space.n, input_dims=(4,84,84) )

if LOAD_CHECKPOINT:
    agent.load_models( SAVE_FILE )


best_score = -np.inf
best_frames = []

max_tries = 400
it = 0

pbar = tqdm( total = max_tries )
while best_score < 5000 and it <= max_tries:
    pbar.update( 1 )
    it += 1
    done = False
    observation = env.reset()
    score = 0
    
    while not done:
        action = agent.choose_action( observation, 0, eval_flag=evaluation )
        observation_, reward, done, info = env.step( action )
        score += reward
        observation = observation_

    if score > best_score:
        best_score = score
        best_frames = env.get_all_frames()
    #    print("New best score: %.2f" % score )
    pbar.set_description(" Best score %.2f - current score %.2f " % ( best_score, score ) )


print("Final score: %.2f" % best_score)
write_video = True
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
while True:
    f = False
    if write_video:
        out = cv2.VideoWriter( 'MsPacman.avi', fourcc, 30, (320, 420) )

    for frame in best_frames:
        h, w, c = frame.shape
        frame = cv2.resize( frame, (w*2, h*2) )
        if write_video:
            out.write( frame[...,::-1] )
        cv2.imshow('output', frame[...,::-1])
        if cv2.waitKey(1000//30) & 0xFF==ord('q'):
            f = True
            break
    if write_video:
        write_video = False
        print("DONE!")
        out.release()
    if f:
        break

cv2.destroyAllWindows()

