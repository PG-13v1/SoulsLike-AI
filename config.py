# config.py — hyperparameters for the complex Soulslike AI boss

ENV_PARAMS = {
    "num_actions": 7,          # more moves = more complex strategy
    "obs_shape": (10,),        # richer state space
}

TRAINING_PARAMS = {
    "episodes": 200_000,
    "batch_size": 128,
    "gamma": 0.99,
    "lr": 3e-4,
    "buffer_size": 200_000,
    "target_update_freq": 1000,
    "min_replay_size": 5000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.9995,
    "save_path": "complex_boss_model.pt",
}

MODEL_PARAMS = {
    "hidden_sizes": [256, 256, 128, 64],
    "dropout": 0.1,
    "use_batch_norm": True,
}
