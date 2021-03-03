from collections import deque 
import numpy as np


class ExperienceMemory(object):
    def __init__(self, max_memory_length, history = 4):
        self.history          = history
        self.state_memory     = np.zeros([max_memory_length, 84, 84, self.history], dtype=np.uint8)#deque(maxlen=max_memory_length)
        self.action_memory    = np.zeros([max_memory_length], dtype=np.uint8)#deque(maxlen=max_memory_length)
        self.reward_memory    = np.zeros([max_memory_length], dtype=np.float16)#deque(maxlen=max_memory_length)
        self.done_memory      = np.zeros([max_memory_length], dtype=np.uint8)#deque(maxlen=max_memory_length)
        self.counter          = 0
        self.ptr              = 0
        self.max_memory_length= max_memory_length

    def get_length(self):
        return self.counter#len(self.state_memory)

  # Store a new memory
    def store_transition(self, state, action, reward, done):
        if state.dtype == np.float32:
            self.state_memory[self.ptr % self.max_memory_length ] = (state*255.).astype(np.uint8)
        else:
            self.state_memory[self.ptr % self.max_memory_length ] = state
        self.action_memory[self.ptr % self.max_memory_length] = action
        self.reward_memory[self.ptr % self.max_memory_length] = reward
        self.done_memory[self.ptr % self.max_memory_length]   = int(done)
        self.ptr = (self.ptr + 1) % self.max_memory_length
        self.counter = min( self.counter+1, self.max_memory_length ) 

    def store_multiple_transitions( self, states, actions, rewards, dones):
        for ind in range(states.shape[0]):
            self.store_transition( states[ind], actions[ind], rewards[ind], dones[ind] )


  # Get out batch_size samples from the memory
    def sample_buffer(self, batch_size):
        states      = []
        actions     = []
        rewards     = []
        next_states = []
        dones       = []

        # choosable_inds = list( range( 0, self.ptr-1)) + list(range(self.ptr, self.counter))
        # choosable_inds = list(set( choosable_inds ))
        prob_choose_inds = np.ones([self.counter], dtype=np.float32)
        prob_choose_inds[self.ptr-1] = 0
        prob_choose_inds = prob_choose_inds / np.sum( prob_choose_inds)

        sample_id   = np.random.choice( self.counter, size=[batch_size], replace=False, p=prob_choose_inds )
        states      = self.state_memory[sample_id % self.counter].astype(np.float32) / 255.
        actions     = self.action_memory[sample_id % self.counter].astype(np.int32)
        rewards     = self.reward_memory[sample_id % self.counter].astype(np.float32)
        next_states = self.state_memory[(sample_id+1) % self.counter].astype(np.float32) / 255.
        dones       = self.done_memory[sample_id % self.counter].astype(np.int32)

        return states, actions, rewards, next_states, dones



if __name__ == '__main__':
    from tqdm import tqdm
    test_size = 512000
    e = ExperienceMemory(test_size)
    pbar = tqdm(total = test_size)
    while (e.get_length() < test_size):
        pbar.update(1)
        e.store_transition( np.random.randn( 84, 84, 4 ), 1, 0., 0 )

    print("Memory should be full now")

    input("Press enter to continue")
    for i in range(10000):
        pbar.update(1)
        e.store_transition( np.random.randn( 84, 84, 4 ), 1, 0., 0 )
        s = e.sample_buffer(32)
    print("Done!")
