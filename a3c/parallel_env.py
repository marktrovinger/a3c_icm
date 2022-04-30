import torch.multiprocessing as mp

class ParallelEnv:
    def __init__(self, worker, env_id, num_threads):
        thread_names = [str(i) for i in range(num_threads)]

        self.procs = [mp.Process(target=worker, args=(name, env_id))
                      for name in thread_names]
        [p.start() for p in self.procs]
        [p.join() for p in self.procs]
