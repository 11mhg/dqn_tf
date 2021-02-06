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
done = False
observation = env.reset()
score = 0

while not done:
    action = agent.choose_action( observation, 0, eval_flag=evaluation )
    observation_, reward, done, info = env.step( action )
    score += reward
    observation = observation_

print("Final score: %.2f" % score)
while True:
    f = False
    allFrames = env.get_all_frames()
    for frame in allFrames:
        cv2.imshow('output', frame[...,::-1])
        if cv2.waitKey(1000//30) & 0xFF==ord('q'):
            f = True
            break
    if f:
        break

cv2.destroyAllWindows()

