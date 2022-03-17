import gym

class RepeatAction(gym.Wrapper):
    def __init__(self, env, repeat, fire_first) -> None:
        super(RepeatAction, self).__init__(env)
        self.env = env
        self.repeat = repeat
        self.first_first = fire_first

    def step(self, action):
        total_reward = 0
        done = False
        state = None
        info = None
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            state = obs
            info = info
            total_reward = total_reward + reward
            if done:
                break
        return state, total_reward, done, info

    def reset(self):
        state = self.env.reset()
        if self.first_first:
            # we need to pass an action
            state = self.env.step()
        return state 



class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, new_shape):
        super(PreprocessFrame, self).__init__(env)
        self.env = env
        self.shape = new_shape
