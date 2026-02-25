SoulsLike-AI

Welcome to SoulsLike-AI, an AI-driven project exploring intelligent agent behavior in a Souls-like game environment.
(This repository currently has minimal metadata — you can customize this README further to match your exact goals and functionality.) 
GitHub

📌 Project Overview

SoulsLike-AI is a Python-based project that implements reinforcement learning (RL) techniques to train and evaluate AI agents for challenges inspired by Souls-like game mechanics — known for strategic combat, difficult enemies, and tactical decision-making. 
GitHub

“Souls-like” games are an action RPG subgenre defined by high difficulty, pattern recognition, tactical combat, and environmental exploration. 
Wikipedia

🚀 Features

✔️ Reinforcement Learning agent implementations
✔ Python simulation environment for training
✔ Visualizations of training performance
✔ Modular project structure for experimentation

Note: You can extend this for additional experiments such as neural network policy learning, curriculum learning, or boss AI benchmarking.

🗂 Project Structure

Here are the main files included in the repo: 
GitHub

SoulsLike-AI/
├── config.py                # Configuration parameters
├── dqn_agent.py             # DQN agent implementation
├── environment.py           # Game/environment simulator
├── main.py                  # Entry point for training/testing
├── replay_buffer.py         # Experience replay buffer
├── traditional_boss.py      # Boss environment logic
├── train_ai_boss.py         # Training script for boss AI
└── visualize_rewards.py     # Reward/metrics visualization

🛠 Tech Stack

This project is built with:

Python 3.x

Reinforcement Learning Algorithms

Popular Python libraries (e.g., NumPy, Matplotlib, TensorFlow/PyTorch if added)

(Install any additional dependencies you use in requirements.txt — if missing, consider adding one.)

📦 Installation

Clone the repository

git clone https://github.com/PG-13v1/SoulsLike-AI.git
cd SoulsLike-AI


Create & activate a virtual environment

python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt


(If you don’t have a requirements.txt yet, generate it with: pip freeze > requirements.txt after installing libs.)

📊 Usage
🧠 Train Agent
python train_ai_boss.py

🕹 Run Evaluation / Visualize Performance
python visualize_rewards.py

🔧 Custom Configuration

Edit config.py to change:

Learning rates

Exploration settings

Environment parameters

Training episodes


🌐 **Web-Based Dashboard (Local)**

You can view the saved plots and graphs in a simple local website.

1. Ensure you have Flask installed (`Flask` is already included in `requirements.txt`).
2. Run the dashboard server:

```bash
python web_dashboard.py
```

3. Open your browser at `http://127.0.0.1:5000/`.
4. Use the **Regenerate plots** link to re-create any static images from the logs (they are stored under `visualizations/`).

    The homepage also includes links to dynamically generated charts:
    * `/action_distribution.png` – boss action usage histogram
    * `/policy_confidence.png` – smoothed max-Q values over training
    * `/decision_explanation.png` – Q‑values for a sample decision
    * `/state_tsne.png` – t‑SNE projection of recorded states

    Those endpoints render a PNG on the fly so you can inspect them without
    first saving a file.

The site will list any `.png` files located in the `visualizations` folder. You can click an image to open it in a new tab.

> ⚠️ The dashboard currently shows static files. If you need additional plots (e.g. action distributions or TSNE), run the corresponding functions manually or extend the Flask app with new endpoints.

🎯 Goals & Roadmap

Potential ways to evolve this project:

✅ Train smarter agents using advanced RL methods
✅ Add more Souls-like environment mechanics
✅ Integrate neural network policies (e.g., DQN with CNNs)
✅ Create benchmarks vs. human players

📄 About Souls-like Games

The name Souls-like refers to an action RPG subgenre inspired by games such as Dark Souls, Bloodborne, and Elden Ring, known for high difficulty and strategic combat design. 
Wikipedia

📫 Contributing

Feel free to open issues or submit pull requests! If you want to develop new AI agents or environment enhancements, create feature branches and submit PRs for review.

🧡 License

Add a license to clarify terms (e.g., MIT / Apache 2.0). If not yet included, consider adding one to your repo.
