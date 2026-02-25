import torch
import numpy as np
from environment import ComplexSoulslikeEnv
from dqn_agent import DQNAgent
from config import TRAINING_PARAMS
from tqdm import trange
import matplotlib.pyplot as plt
import json


def train_ai_boss():
    env = ComplexSoulslikeEnv(adaptive=True)
    agent = DQNAgent()
    episode_rewards = []
    log_data=[]

    # Print GPU info
    print(f"Using device: {agent.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    for ep in trange(TRAINING_PARAMS["episodes"], desc="Training AI Boss"):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            
            # Get q_values for logging
            agent.q_net.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.q_net(state_tensor).cpu().numpy().flatten()
            agent.q_net.train()
            
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            loss = agent.update(TRAINING_PARAMS["batch_size"], TRAINING_PARAMS["gamma"])
            state = next_state
            total_reward += reward

            log_data.append({
                "episode": ep,
                "state": state.tolist(),
                "action": action,
                "reward": reward,
                "next_state": next_state.tolist(),
                "done": done,
                "loss": loss,
                "q_values": q_values.tolist()
            })

        episode_rewards.append(total_reward)

        if ep % 1000 == 0 and ep > 0:
            torch.save(agent.q_net.state_dict(), TRAINING_PARAMS["save_path"])

    with open("logs/ai_decision_log.json", "w") as f:
        json.dump(log_data, f)

    plt.plot(np.convolve(episode_rewards, np.ones(500)/500, mode='valid'))
    plt.title("AI Boss Reward Progress (Smoothed)")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

    np.save("logs/episode_rewards.npy", episode_rewards)
    torch.save(agent.q_net.state_dict(), TRAINING_PARAMS["save_path"])
    print("Training Complete — Model Saved ✅")
