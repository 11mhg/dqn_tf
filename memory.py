from collections import deque 
import numpy as np


class ExperienceMemory(object):
  def __init__(self, max_memory_length):
      self.state_memory     = np.zeros([max_memory_length, 4, 84, 84], dtype=np.uint8)#deque(maxlen=max_memory_length)
      self.action_memory    = np.zeros([max_memory_length], dtype=np.uint8)#deque(maxlen=max_memory_length)
      self.reward_memory    = np.zeros([max_memory_length], dtype=np.float16)#deque(maxlen=max_memory_length)
      self.done_memory      = np.zeros([max_memory_length], dtype=np.uint8)#deque(maxlen=max_memory_length)
      self.counter          = 0
      self.max_memory_length= max_memory_length

  def get_length(self):
      return self.counter#len(self.state_memory)

  # Store a new memory    
  def store_transition(self, state, action, reward, done):
      self.state_memory[self.counter % self.max_memory_length ] = (state*255.).astype(np.uint8)
      self.action_memory[self.counter % self.max_memory_length] = action
      self.reward_memory[self.counter % self.max_memory_length] = reward
      self.done_memory[self.counter % self.max_memory_length]   = done
      self.counter = min( self.counter+1, self.max_memory_length ) 

  # Get out batch_size samples from the memory
  def sample_buffer(self, batch_size):
    states      = []
    actions     = []
    rewards     = []
    next_states = []
    dones       = []
  
    for i in range(batch_size):
      sample_id  = np.random.randint(0, self.counter-1)
      states.append( self.state_memory[sample_id].astype(np.float32) / 255. )
      actions.append(self.action_memory[sample_id])
      rewards.append(self.reward_memory[sample_id])
      next_states.append( self.state_memory[sample_id+1].astype(np.float32) / 255. )
      dones.append(self.done_memory[sample_id])

    return np.asarray(states), np.asarray(actions), np.asarray(rewards), np.asarray(next_states), np.asarray(dones)




if __name__ == '__main__':
    from tqdm import tqdm
    e = ExperienceMemory(64000)
    pbar = tqdm(total = 64000)
    while (e.get_length() < 64000):
        pbar.update(1)
        e.store_transition( np.random.randn( 4, 84, 84 ), 1, 0., 0 )

    print("Memory should be full now")

    input("Press enter to continue")
    for i in range(10000):
        pbar.update(1)
        e.store_transition( np.random.randn( 4, 84, 84 ), 1, 0., 0 )


    print("Done!")
