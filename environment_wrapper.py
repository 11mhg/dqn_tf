import tensorflow as tf
import numpy as np
import gym
import cv2

FRAME_HEIGHT = 84
FRAME_WIDTH  = 84


class HelperEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(HelperEnv, self).__init__(env)
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
    env = HelperEnv(env)
    return env

if __name__=="__main__":
    env = make_env('MsPacman-v0')
    obs = env.reset()

    done = False
    action = env.action_space.sample()
    while not done:
        obs, reward, done, _ = env.step( action )
        action = env.action_space.sample()
        cv2.imshow('o', obs )
        if cv2.waitKey(1000//60) & 0xFF==ord('q'):
            break
    
    cv2.destroyAllWindows()

    allFrames = env.get_all_frames()

    for frame in allFrames:
        cv2.imshow('output', frame[...,::-1] )
        if cv2.waitKey(1000//60) & 0xFF==ord('q'):
            break
    cv2.destroyAllWindows()
    
