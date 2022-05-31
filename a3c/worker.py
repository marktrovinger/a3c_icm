import numpy as np
import gym
from actor_critic import ActorCritic
from wrappers.wrappers import make_env
from memory import Memory
import torch as T

def worker(name, env_id, global_agent, optimizer, global_idx, n_actions, input_shape, n_threads):
    #env = gym.make(env_id)
    T_MAX = 20
    
    
    local_agent = ActorCritic(n_actions=n_actions, input_dims=input_shape)
    memory = Memory()

    # swap channels in our input
    frame_buffer = [input_shape[1], input_shape[2], 1]
    # create the env, using make_env function
    env = make_env(env_id, shape = frame_buffer)

    # add time_steps to variables, increase max_eps
    episode, max_eps, scores = 0, 10, []
    # we don't need to run very many episodes
    while episode < max_eps:
        obs = env.reset()
        score = 0
        done = False
        episode_steps = 0
        # set hx to 0
        hx = T.zeros(1, 256)
        while not done:
            # convert state to PyTorch tensor
            state = T.tensor([obs], T.float)
            # pass state and hx to local_agent
            action, value, log_probs, hx = local_agent(state, hx)
            obs_, reward, done, _ = env.step(action)
            # save memory
            memory.store_memory(log_probs, value, reward)
            score += reward
            obs = obs_
            # increment steps
            episode_steps += 1
            # perform loss and gradient calculations
            if episode_steps % T_MAX == 0:
                rewards, values, probs = memory.sample_memory()
                loss = local_agent.calc_cost(obs, hx, done, rewards, values, probs)
                # zero gradients
                optimizer.zero_grad()
                # detach hx
                hx = hx.detach_()
                # clip gradients
            # sync local with global
            # step optimizer
            # clear memory
        # increment episode
        # score calculations
        scores.append(score)
        print(f'Episode: {episode}, Process: {name}, score: {score}')
        episode += 1
    env.close()