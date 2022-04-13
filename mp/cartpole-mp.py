import os
import torch.multiprocessing as mp
import gym

OMS_NUM_THREADS = '1'

class ParallelEnv:
    def __init__(self, env_id, num_threads):
        thread_names = [str(i) for i in range(num_threads)]

        self.procs = [mp.Process(target=worker, args=(name, env_id))
                      for name in thread_names]
        [p.start() for p in self.procs]
        [p.join() for p in self.procs]

# create a function that initializes and runs cartpole
def worker(name, env_id):
    env = gym.make(env_id)
    episode, max_eps, scores = 0, 10, [] 

    # we don't need to run very many episodes
    while episode < max_eps:
        obs = env.reset()
        score = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs_, reward, done, _ = env.step(action)
            score += reward
            obs = obs_
        scores.append(score)
        print(f'Episode: {episode}, Process: {name}, score: {score}')
        episode += 1
    env.close()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    env_id = 'CartPole-v0'
    n_threads = 4
    env = ParallelEnv(env_id=env_id, num_threads=n_threads)
