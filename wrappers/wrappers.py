import gym

class RepeatAction(gym.Wrapper):
    def __init__(self, env, repeat, fire_first) -> None:
        super(RepeatAction, self).__init__(env)
        self.env = env
        self.repeat = repeat
        self.first_first = fire_first

    def step(self, action):
        pass



class PreprocessFrame: