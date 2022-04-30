import numpy as np
import gym

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