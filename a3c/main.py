from parallel_env import ParallelEnv
import torch.multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn')
    env_id = 'CartPole-v0'
    n_threads = 4
    env = ParallelEnv(env_id=env_id, num_threads=n_threads)