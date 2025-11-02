import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from replay_buffer import ReplayBuffer
from config import TRAINING_PARAMS, MODEL_PARAMS, ENV_PARAMS

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim

        # First layer (no batch norm)
        layers.append(nn.Linear(prev_dim, MODEL_PARAMS["hidden_sizes"][0]))
        layers.append(nn.LeakyReLU())
        if MODEL_PARAMS["dropout"] > 0:
            layers.append(nn.Dropout(MODEL_PARAMS["dropout"]))
        prev_dim = MODEL_PARAMS["hidden_sizes"][0]
        
        # Hidden layers with optional batch norm
        for hidden_size in MODEL_PARAMS["hidden_sizes"][1:]:
            if MODEL_PARAMS["use_batch_norm"]:
                layers.append(nn.BatchNorm1d(prev_dim))
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.LeakyReLU())
            if MODEL_PARAMS["dropout"] > 0:
                layers.append(nn.Dropout(MODEL_PARAMS["dropout"]))
            prev_dim = hidden_size
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the input
        return self.net(x)

class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obs_dim = ENV_PARAMS["obs_shape"][0]
        act_dim = ENV_PARAMS["num_actions"]

        self.q_net = QNetwork(obs_dim, act_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, act_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=TRAINING_PARAMS["lr"])
        self.replay_buffer = ReplayBuffer(TRAINING_PARAMS["buffer_size"])
        self.steps = 0
        self.epsilon = TRAINING_PARAMS["epsilon_start"]

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(ENV_PARAMS["num_actions"])
        
        # Set network to eval mode for inference
        self.q_net.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state)
        # Set back to train mode
        self.q_net.train()
        return int(torch.argmax(q_values).item())

    def update(self, batch_size, gamma):
        if len(self.replay_buffer) < TRAINING_PARAMS["min_replay_size"]:
            return 0
        
        self.q_net.train()
        self.target_net.eval()
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay and target update
        self.epsilon = max(TRAINING_PARAMS["epsilon_end"], self.epsilon * TRAINING_PARAMS["epsilon_decay"])
        if self.steps % TRAINING_PARAMS["target_update_freq"] == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.steps += 1
        return loss.item()
