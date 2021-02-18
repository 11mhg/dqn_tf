import tensorflow as tf
import numpy as np
import gym
import cv2

FRAME_HEIGHT = 84
FRAME_WIDTH  = 84


class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self._skip = skip
        self.FRAMES = []
    
    def get_all_frames(self):
        return self.FRAMES


    def step(self, action):
        t_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self.FRAMES.append(obs)
            t_reward += reward
            if done:
                break
         # When we are in evaluation mode, we need to send the deepest wrapped 
        # (original) observation back out so that we can see how the agent plays
        return obs, t_reward, done, info

    def reset(self):
        self._obs_buffer = []
        obs = self.env.reset()
        self.FRAMES = [obs]
        self._obs_buffer.append(obs)
        return obs

class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(FRAME_HEIGHT,FRAME_WIDTH,1), dtype=np.uint8)

    def observation(self, obs):
        new_frame = np.reshape(obs, obs.shape).astype(np.float32)
        # convert to grayscale
        new_frame = cv2.cvtColor( new_frame, cv2.COLOR_RGB2GRAY)
        # scale to frame height and width
        new_frame = cv2.resize(new_frame,(FRAME_WIDTH, FRAME_HEIGHT),cv2.INTER_NEAREST)[:, :, np.newaxis]
        # convert to numpy array
        new_frame = np.asarray(new_frame)
        return new_frame.astype(np.float32)

class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super(MoveImgChannel, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                            shape=(self.observation_space.shape[-1],
                                   self.observation_space.shape[0],
                                   self.observation_space.shape[1]),
                            dtype=np.float32)
  
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class NormalizeFrame(gym.ObservationWrapper):
    # The match is easier if everything is normalized to be betwee 0 and 1.
    def observation(self, obs):
        return np.clip( np.array(obs).astype(np.float32) / 255.0, 0., 1. )

class FrameStacker(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(FrameStacker, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                             env.observation_space.low.repeat(n_steps, axis=0),
                             env.observation_space.high.repeat(n_steps, axis=0),
                             dtype=np.float32)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

# Apply all the wrappers on top of each other.
def make_env(env_name):
    env = gym.make(env_name)
    env = SkipEnv(env)
    env = PreProcessFrame(env)
    env = MoveImgChannel(env)
    env = FrameStacker(env, 4)
    return NormalizeFrame(env)



if __name__=="__main__":
    env = make_env('MsPacman-v0')
    obs = env.reset()

    done = False
    action = env.action_space.sample()
    while not done:
        obs, reward, done, _ = env.step( action )
        action = env.action_space.sample()
        cv2.imshow('o', np.transpose(obs, [1,2,0])[...,:3] )
        if cv2.waitKey(1000//60*4) & 0xFF==ord('q'):
            break
    
    cv2.destroyAllWindows()

    allFrames = env.get_all_frames()

    for frame in allFrames:
        cv2.imshow('output', frame[...,::-1] )
        if cv2.waitKey(1000//30) & 0xFF==ord('q'):
            break
    cv2.destroyAllWindows()
    
