import os
import torch.multiprocessing as mp
import gym

class ParallelEnv:
    def __init__(self, env_id, num_threads):
        thread_names = [str(i) for i in range(num_threads)]

        self.procs = [mp.Process(target=cartpole_mp, args=(name, env_id))
                      for name in thread_names]
        [p.start() for p in self.procs]
        [p.join() for p in self.procs]

# create a function that initializes and runs cartpole
def cartpole_mp(name, env_id):
    env = gym.make(env_id)

    env.reset()

    # we don't need to run very many episodes
    for episode in range(episodes):
        obs, reward, done, _ = env.step(env.action_space.sample())
        print(f'Process: {os.getpid()}, score: {reward}')

    env.close()

if __name__ == '__main__':
    with Pool(5) as p:
        p.map(cartpole_mp, [5])

