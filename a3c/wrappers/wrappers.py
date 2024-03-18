import gymnasium as gym
import numpy as np
import cv2
from collections import deque
from gymnasium.experimental.wrappers import AtariPreprocessingV0
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
#from stable_baselines3.common.env_util import make_atari_env
#from stable_baselines3.common.atari_wrappers import AtariWrapper

class RepeatAction(gym.Wrapper):
    def __init__(self, env = None, repeat = 4, fire_first = False):
        super(RepeatAction, self).__init__(env)
        self.env = env
        self.repeat = repeat
        self.fire_first = fire_first

    def step(self, action):
        total_reward = 0.0
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward = total_reward + reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self):
        state = self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            # we need to pass an action
            state, _, _, _ = self.env.step(1)
        return state 


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, new_shape):
        super(PreprocessFrame, self).__init__(env)
        self.env = env
        self.shape = (new_shape[2], new_shape[0], new_shape[1])
        # swap channel axis in new shape
        self.observation_space = gym.spaces.Box(low = 0.0, high = 1.0, shape = self.shape,
                                                dtype = np.float32)
        # set observation space to new shape

    def observation(self, observation):
        # convert to grey-scale
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # resize observation to new shape
        resized = cv2.resize(observation, self.shape[1:], interpolation=cv2.INTER_AREA)
        # convert to numpy array
        new_resized = np.array(resized, dtype=np.int8).reshape(self.shape)
        # move channel axis from 2 to 0
        new_resized = new_resized / 255.0
        # observation /= 255
        # return observation
        return new_resized

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env = None, repeat=4):
        super(StackFrames, self).__init__(env)
        self.frame_stack = deque(maxlen=repeat)
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(repeat, axis=0),
                                                env.observation_space.high.repeat(repeat, axis=0),
                                                dtype=np.float32)


    def reset(self):
        self.frame_stack.clear()
        obs = self.env.reset()
        for i in range(self.frame_stack.maxlen):
            self.frame_stack.append(obs)
        frame_stack = np.array(self.frame_stack).reshape(self.observation_space.low.shape)
        return frame_stack

    def observation(self, observation):
        self.frame_stack.append(observation)
        frame_stack = np.array(self.frame_stack)
        frame_stack.reshape(self.observation_space.low.shape)
        return frame_stack

def make_env(env_name, new_shape = (42,42,1), frame_repeat = 4):
    env = gym.make(env_name)
    #env = RepeatAction(env)
    #env = StackFrames(env, repeat=frame_repeat)
    #env = PreprocessFrame(env, new_shape=new_shape)
    #env = AtariWrapper(env_name, screen_size=42#)

    env = AtariPreprocessing(env, screen_size=new_shape[0], frame_skip = frame_repeat)
    return env
