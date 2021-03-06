import torch.multiprocessing as mp
from worker import worker
from actor_critic import ActorCritic
from shared_adam import SharedAdam

class ParallelEnv:
    def __init__(self, n_actions, input_shape, global_idx, env_id, num_threads):
        thread_names = [str(i) for i in range(num_threads)]
        global_agent = ActorCritic(n_actions=n_actions, input_dims=input_shape)
        global_agent.share_memory()
        global_optimizer = SharedAdam(global_agent.parameters(), lr=1e-4)

        self.procs = [mp.Process(target=worker, args=(name, env_id, global_agent, global_optimizer, global_idx, 
                                                    n_actions, input_shape, num_threads))
                      for name in thread_names]
        [p.start() for p in self.procs]
        [p.join() for p in self.procs]
