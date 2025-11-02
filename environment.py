import gym
import numpy as np
from gym import spaces
from config import ENV_PARAMS

class ComplexSoulslikeEnv(gym.Env):
    """
    Complex environment for a Soulslike boss AI.
    The state includes player and boss stats, aggression levels, and randomness.
    """
    def __init__(self, adaptive=True):
        super().__init__()
        self.action_space = spaces.Discrete(ENV_PARAMS["num_actions"])
        self.observation_space = spaces.Box(low=0, high=1, shape=ENV_PARAMS["obs_shape"], dtype=np.float32)
        self.adaptive = adaptive
        self.reset()

    def reset(self):
        self.player_health = 1.0
        self.boss_health = 1.0
        self.player_aggression = np.random.uniform(0.3, 0.7)
        self.state = np.random.rand(*ENV_PARAMS["obs_shape"])
        return self.state

    def step(self, action):
        # Player behavior is semi-random but aggressive
        player_action = np.random.randint(0, self.action_space.n)
        damage = np.random.uniform(0.05, 0.15)

        # Basic combat rules
        if action == player_action:
            self.boss_health -= damage * 0.5
            reward = -1
        else:
            self.player_health -= damage
            reward = 2 if self.adaptive else 1

        # Encourage diversity in boss actions (varied moveset)
        diversity_bonus = np.random.choice([0, 0.2])
        reward += diversity_bonus

        done = self.player_health <= 0 or self.boss_health <= 0
        self.state = np.clip(np.random.rand(*ENV_PARAMS["obs_shape"]) * (self.boss_health + 0.5), 0, 1)
        return self.state, reward, done, {}
