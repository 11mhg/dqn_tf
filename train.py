import os
import time
from multiprocessing import Process, Queue
os.environ['TF_XLA_FLAGS']="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

#import tensorflow as tf
#tf.config.set_soft_device_placement(True)
#from agents import Agent
#from environment_wrapper import make_env
import sys, shutil
import numpy as np
from tqdm import tqdm
num_iter = 0 

ENV_NAME = 'MsPacman-v0'
SAVE_FILE = './' + ENV_NAME + '.h5' 

LOAD_CHECKPOINT = False#True

evaluation = False

SAVE_FREQUENCY = 100000
MODEL_COPY_FREQUENCY = 20000 

BATCH_SIZE = 32 
LEARNING_RATE = 1e-4

PREFILL_LENGTH = 64000


if __name__=='__main__':
    import tensorflow as tf
    from agents import Agent
    from environment_wrapper import make_env


    writer = tf.summary.create_file_writer('./logs/train/' + str(time.time())  )

    num_iter = 0
    env = make_env( ENV_NAME )
    agent = Agent( env= env,
            n_actions=env.action_space.n,
            input_dims=(84,84,4),
            batch_size=BATCH_SIZE,
            memory_length=512000,
            learning_rate=LEARNING_RATE)
    curr_state = agent.reset_env()
    score = 0
    done = False
    best_score = 0
    all_scores = []
    all_losses = []
    all_eps    = []
    number_of_syncs = 0
    pbar = tqdm( total=5e6 )
    num_received = 0

    for _ in range(PREFILL_LENGTH):
        pbar.set_description("Prefilling %d / 64k" %(agent.memory.get_length()))
        curr_state, action, reward, done, info = agent.step( curr_state, 1)
        if done:
            curr_state = agent.reset_env()


    while True:
        pbar.update(1)
        num_iter += 1
        if len(all_scores) > 0 and len(all_losses) > 0 and num_iter % 20 == 0:
            with writer.as_default():
                tf.summary.scalar('lr', LEARNING_RATE, step=num_iter)
                tf.summary.scalar('loss', np.mean(all_losses[-100:]), step=num_iter)
                tf.summary.scalar('score', np.mean(all_scores[-100:]), step=num_iter)
                tf.summary.scalar('epsilon', agent.get_eps( t=num_iter ), step=num_iter)
                tf.summary.scalar('num_syncs', number_of_syncs, step=num_iter)
            pbar.set_description("%d - avg loss: %.4f avg score: %.4f mem length: %d" % (num_iter, 
                    np.mean( all_losses[-100:]), 
                    np.mean( all_scores[-100:]),
                    agent.memory.get_length() ))

        curr_state, action, reward, done, info = agent.step( curr_state, num_iter )

        score += reward

        if done:
            curr_state = agent.reset_env()
            all_scores.append( score )
            if score > best_score:
                best_score = score
                agent.save_models(SAVE_FILE)
            score = 0
            if (num_iter - PREFILL_LENGTH) > 5e6:
                break

        loss = agent.learn()
        if loss is not None:
            all_losses.append(loss)


        if num_iter % (MODEL_COPY_FREQUENCY) == 0 and num_iter > 0:
            agent.target_network.set_weights( agent.q_network.get_weights() )
            number_of_syncs += 1
    print("Done")


