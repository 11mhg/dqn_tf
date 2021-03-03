import tensorflow as tf
from collections import deque
import numpy as np
import gym
import cv2

FRAME_HEIGHT = 84
FRAME_WIDTH  = 84

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs




class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs


class SaveFramesEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(SaveFramesEnv, self).__init__(env)
        self.FRAMES = []
    
    def get_all_frames(self):
        return self.FRAMES

    def step(self, action):
        t_reward = 0.0
        done = False
        obs, reward, done, info = self.env.step(action)
        self.FRAMES.append(obs)
        t_reward += reward
        return obs, t_reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.FRAMES = [obs] 
        return obs

# Apply all the wrappers on top of each other.
def make_env(env_name):
    env = gym.make(env_name)
    env = SaveFramesEnv(env)
    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env)
    env = WarpFrame(env)
    return env

if __name__=="__main__":
    env = make_env('MsPacman-v0')
    obs = env.reset()

    done = False
    action = env.action_space.sample()
    while not done:
        obs, reward, done, _ = env.step( action )
        action = env.action_space.sample()
        cv2.imshow('o', obs[...,::-1] )
        if cv2.waitKey(1000//60) & 0xFF==ord('q'):
            break
    
    cv2.destroyAllWindows()

    allFrames = env.get_all_frames()

    for frame in allFrames:
        cv2.imshow('output', frame[...,::-1] )
        if cv2.waitKey(1000//60) & 0xFF==ord('q'):
            break
    cv2.destroyAllWindows()
    
