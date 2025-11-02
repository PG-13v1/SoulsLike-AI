import torch
import numpy as np
from environment import ComplexSoulslikeEnv
from dqn_agent import DQNAgent
from config import TRAINING_PARAMS
from tqdm import trange
import matplotlib.pyplot as plt

def train_ai_boss():
    env = ComplexSoulslikeEnv(adaptive=True)
    agent = DQNAgent()
    episode_rewards = []

    for ep in trange(TRAINING_PARAMS["episodes"], desc="Training AI Boss"):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            loss = agent.update(TRAINING_PARAMS["batch_size"], TRAINING_PARAMS["gamma"])
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        if ep % 1000 == 0 and ep > 0:
            torch.save(agent.q_net.state_dict(), TRAINING_PARAMS["save_path"])

    plt.plot(np.convolve(episode_rewards, np.ones(500)/500, mode='valid'))
    plt.title("AI Boss Reward Progress (Smoothed)")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

    torch.save(agent.q_net.state_dict(), TRAINING_PARAMS["save_path"])
    print("Training Complete — Model Saved ✅")
