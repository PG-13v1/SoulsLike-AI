# config.py — hyperparameters for the complex Soulslike AI boss
ENV_PARAMS = {
    "num_actions": 7,
    "obs_shape": (10,),
}

TRAINING_PARAMS = {
    "episodes": 50,
    "batch_size": 128,        # smaller batch for faster learning
    "gamma": 0.99,
    "lr": 5e-4,               # slightly higher learning rate
    "buffer_size": 20_000,    # large buffers unnecessary for 50 episodes
    "target_update_freq": 200,
    "min_replay_size": 500,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.95,    # much faster exploration decay
    "save_path": "complex_boss_model.pt",
}

MODEL_PARAMS = {
    "hidden_sizes": [128, 128],  # smaller network for small training run
    "dropout": 0.0,
    "use_batch_norm": False
}
