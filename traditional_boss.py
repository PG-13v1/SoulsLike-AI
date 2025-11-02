import numpy as np
from environment import ComplexSoulslikeEnv

def run_traditional_boss(episodes=100):
    env = ComplexSoulslikeEnv(adaptive=False)
    total_rewards = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = np.random.randint(0, env.action_space.n)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
        total_rewards.append(ep_reward)
    return total_rewards
