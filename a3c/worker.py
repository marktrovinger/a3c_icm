import numpy as np
import gym
from actor_critic import ActorCritic
from wrappers.wrappers import make_env
from memory import Memory

def worker(name, env_id, agent, optimizer, global_idx, n_actions, input_shape, n_threads):
    #env = gym.make(env_id)
    env = make_env(env_id)
    episode, max_eps, scores = 0, 10, []
    
    local_agent = ActorCritic(n_actions=n_actions, input_dims=input_shape)

    memory = Memory()

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