from parallel_env import ParallelEnv
import torch.multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn')
    env_id = 'PongNoFrameskip-v0'
    # 6 for pong, 4 for cartpole
    n_threads = 1
    input_shape = [4, 42, 42]
    n_actions = 6
    global_idx = mp.Value('i', 0)
    env = ParallelEnv(n_actions=n_actions, input_shape=input_shape,
                      global_idx=global_idx, env_id=env_id, num_threads=n_threads)
