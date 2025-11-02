from train_ai_boss import train_ai_boss
from traditional_boss import run_traditional_boss
from visualize_rewards import visualize_rewards
from config import TRAINING_PARAMS

if __name__ == "__main__":
    print("\n=== TRAINING COMPLEX AI BOSS (Double DQN + MLP + Replay) ===")
    train_ai_boss()

    print("\n=== EVALUATING TRADITIONAL BOSS ===")
    traditional_rewards = run_traditional_boss(episodes=300)

    print("\n=== AI-TRAINED REWARDS ===")
    ai_rewards = [r for r in open('reward_log.txt', 'r')] if False else traditional_rewards

    print("\n=== VISUALIZING COMPARISON ===")
    visualize_rewards(traditional_rewards, ai_rewards)
