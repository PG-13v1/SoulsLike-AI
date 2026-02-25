from train_ai_boss import train_ai_boss
from traditional_boss import run_traditional_boss
from config import TRAINING_PARAMS
from visualizations.dashboard import plot_learning_curve, plot_action_distribution, plot_policy_confidence, plot_decision_explanation, plot_state_space_tsne,visualize_rewards
import numpy as np

if __name__ == "__main__":
    print("\n=== TRAINING COMPLEX AI BOSS (Double DQN + MLP + Replay) ===")
    train_ai_boss()

    print("\n=== EVALUATING TRADITIONAL BOSS ===")
    traditional_rewards = run_traditional_boss(episodes=300)

    print("\n=== AI-TRAINED REWARDS ===")
    ai_rewards = np.load("logs/episode_rewards.npy")

    print("\n=== VISUALIZING COMPARISON ===")
    visualize_rewards(traditional_rewards, ai_rewards)
    plot_learning_curve()
    plot_action_distribution()
    plot_policy_confidence()
    plot_decision_explanation()
    plot_state_space_tsne()
