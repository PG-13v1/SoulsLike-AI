import matplotlib.pyplot as plt
import numpy as np

def visualize_rewards(traditional_rewards, ai_rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(np.convolve(traditional_rewards, np.ones(5)/5, mode='valid'),
             label='Traditional Boss', color='red', alpha=0.6)
    plt.plot(np.convolve(ai_rewards, np.ones(5)/5, mode='valid'),
             label='AI-Trained Boss (Complex DQN)', color='green', alpha=0.8)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Reward Comparison: Traditional vs Complex AI Boss')
    plt.legend()
    plt.grid(True)
    plt.show()
