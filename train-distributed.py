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
NUM_GENERATORS = 1 

def generate_samples(_id, data_q, out_q, done_queue): #starting_ind = 0, batch_size = BATCH_SIZE, num_to_play = MODEL_COPY_FREQUENCY ):
    #print(_id, 'starting...')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    from agents import Agent
    from environment_wrapper import make_env

    #print(_id, "Ready to rock and roll")
    DONE_MAJOR_LOOP=False
    while not DONE_MAJOR_LOOP:
        _info   = data_q.get(block=True, timeout=60*30) # wait for a max of 30 minutes on timeout
        weights = _info[0]
        starting_ind = _info[1]
        batch_size = _info[2]
        num_to_play = _info[3]

        num_iter = starting_ind
        max_iter = num_iter + num_to_play + 1
        env = make_env( ENV_NAME )
        agent = Agent( env= env,
                n_actions=env.action_space.n,
                input_dims=(84,84,4),
                batch_size=batch_size,
                memory_length= num_to_play+1 )
        curr_state = agent.reset_env()
        score = 0
        done = False
        all_scores = []
    
    
        agent.q_network.set_weights( weights )
        agent.target_network.set_weights( weights )
    
    
        while num_iter < max_iter:
            num_iter += 1    
    
            curr_state, action, reward, done, info = agent.step( curr_state, num_iter )
            
            score += reward
    
            if done:
                curr_state = agent.reset_env()
                all_scores.append( score )
                score = 0

        out = [ [   agent.memory.state_memory, 
                    agent.memory.action_memory,
                    agent.memory.reward_memory,
                    agent.memory.done_memory ],
                all_scores,
                num_iter ]
    
        out_q.put( out, block=False, timeout=None )

        if not done_queue.empty():
            DONE_MAJOR_LOOP = done_queue.get(block=False, timeout=None)


if __name__=='__main__':

    data_qs = []
    out_qs  = []
    done_qs = []

    processes = []
    for _id in range(NUM_GENERATORS):
        data_q = Queue()
        out_q  = Queue()
        done_q = Queue()
        data_qs.append(data_q)
        out_qs.append(out_q)
        done_qs.append(done_q)
        p = Process(target=generate_samples, args=( _id, data_q, out_q, done_q ))
        p.start()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    all_scores = []
    all_losses = []
    all_eps    = []
    number_of_syncs = 0

    pbar = tqdm( total=5e6 )
    num_received = 0
    weights = agent.q_network.get_weights()
    for data_q in data_qs:
        data_q.put([ weights, 
                     num_iter, 
                     BATCH_SIZE, 
                     4000 ], 
                block=False, timeout=None )


    while num_iter < 5e5:
        for data_q, out_q in zip(data_qs, out_qs):
            if not out_q.empty():
                num_received += 1
                _out = out_q.get( block=True, timeout=None )
                agent.memory.store_multiple_transitions( *_out[0] )
                all_scores = all_scores + _out[1]
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
                else:
                    pbar.set_description("%d - Prefilling mem length: %d" % (num_iter, agent.memory.get_length()))
                
                weights = agent.q_network.get_weights()
                data_q.put([ weights, 
                             num_iter, 
                             BATCH_SIZE, 
                             4000 ], 
                             block=False, timeout=None )
                all_scores = all_scores[-100:]
                all_losses = all_losses[-100:]
            
            if agent.memory.get_length() >= 64000:
                loss = agent.learn()
                if loss is not None:
                    all_losses.append(loss)
        
        if agent.memory.get_length() >= 64000:
            pbar.update(4)
            num_iter += 4

        if num_iter % (MODEL_COPY_FREQUENCY) == 0 or (num_iter < 64000 and agent.memory.get_length() < 64000 and \
                num_iter % (MODEL_COPY_FREQUENCY//2) == 0):

            if num_received != len(out_qs):
                pbar.set_description( '%d - Synchronizing agents.... mem length %d' %(num_iter, agent.memory.get_length()) )
                for data_q, out_q in zip(data_qs, out_qs):
                    _out = out_q.get( block=True, timeout=None )
                    agent.memory.store_multiple_transitions( *_out[0] )
                    all_scores = all_scores + _out[1] 

                    weights = agent.q_network.get_weights()
                    data_q.put([ weights,
                                 num_iter,
                                 BATCH_SIZE,
                                 4000 ],
                                 block=False, timeout=None )

            num_received = 0
            agent.target_network.set_weights( agent.q_network.get_weights() )
            number_of_syncs += 1


        if len(all_scores) > 0 and len(all_losses) > 0:
            with writer.as_default():
                tf.summary.scalar('loss', np.mean(all_losses[-100:]), step=num_iter)
                tf.summary.scalar('score', np.mean(all_scores[-100:]), step=num_iter)
                tf.summary.scalar('epsilon', agent.get_eps( t=num_iter ), step=num_iter)
                tf.summary.scalar('num_syncs', number_of_syncs, step=num_iter)
            pbar.set_description("%d - avg loss: %.4f avg score: %.4f mem length: %d" % (num_iter, 
                            np.mean( all_losses[-100:]), 
                            np.mean( all_scores[-100:]),
                            agent.memory.get_length() ))

    for done_q in done_qs:
        done_q.put( True, block=False, timeout=None )


    for p in processes:
        p.join()

   


    #env = make_env( ENV_NAME )    
    #learning_agent = Agent(env = env, 
    #                n_actions=env.action_space.n, 
    #                input_dims=(84,84,4), 
    #                batch_size= BATCH_SIZE )
    #
    #if LOAD_CHECKPOINT:
    #    try:
    #        learning_agent.load_models( SAVE_FILE )
    #    except Exception:
    #        print("Model not found...")
    #
    #learning_agent.save_models( SAVE_FILE ) 
    

#while True:
#    num_episodes+=1 
#    done = False
#    curr_state = agent.reset_env()
#    score = 0
#    while not done:
#        if num_iter > 5e6:
#            break
#        pbar.update(1)
#        num_iter += 1
#
#        curr_state, action, reward, done, info = agent.step( curr_state, num_iter )
#        score += reward
#        l = agent.learn()
#
#        if l is not None:
#            losses.append( l )
#
#        if num_iter % MODEL_COPY_FREQUENCY == 0:
#            pbar.set_description("Updating target_network weights.")
#            agent.target_network.set_weights( agent.q_network.get_weights() )
#
#    scores.append(score)
#    avg_score = np.mean( scores[-100:] )
#    avg_losses = np.mean( losses[-100:] )
#
#    if avg_score > best_score or num_iter % SAVE_FREQUENCY == 0:
#        best_score = avg_score
#        agent.save_models( SAVE_FILE )
#
#    if len(losses) > 0:
#        with writer.as_default():
#            tf.summary.scalar("loss", avg_losses, step=num_iter)
#            tf.summary.scalar("score", avg_score, step=num_iter)
#            tf.summary.scalar('eps', agent.get_eps(num_iter), step=num_iter)
#
#        pbar.set_description(
#            '%d - l: %.4f score: %d average score: %.3f, eps %.2f ' % (num_episodes,
#                    avg_losses, score, avg_score, agent.get_eps(num_iter) )
#        )
#    if num_iter > 5e6:
#        break
#
#
