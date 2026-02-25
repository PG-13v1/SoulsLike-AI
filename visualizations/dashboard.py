"""Utility functions for generating plots used by the web dashboard and
training scripts.

All functions accept a ``save_path`` argument (default ``None``) which will
cause the figure to be saved there instead of shown interactively.  The public
API mirrors what the original ``main.py`` expected.

Plots rely on data logged during ``train_ai_boss``:

* ``logs/episode_rewards.npy`` – array of total reward per training episode
* ``logs/ai_decision_log.json`` – list of step dictionaries containing
  ``state``, ``action`` and ``q_values`` (among other fields)

The ``generate_all`` helper simply calls every plotting function and writes
PNG files into the supplied directory (``visualizations`` by default).
"""

import json
import os
from typing import List, Optional, Union

# make sure matplotlib uses a non-interactive backend so the Flask server
# can create figures from worker threads without popping up a GUI or
# emitting scary warnings.  ``Agg`` works well for PNG output.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.manifold import TSNE
except ImportError:  # type: ignore
    TSNE = None  # used guard in tsne plot


LOG_DIR = "logs"
DEFAULT_VIS_DIR = "visualizations"


# ---- helpers --------------------------------------------------------------

def _load_log_data() -> List[dict]:
    path = os.path.join(LOG_DIR, "ai_decision_log.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _maybe_save_or_show(fig: plt.Figure, save_path: Optional[Union[str, "BytesIO"]]) -> None:
    """Save to ``save_path`` if provided, else display interactively.

    ``save_path`` is normally a filesystem path but ``_fig_response`` passes a
    :class:`io.BytesIO` buffer so that charts can be streamed back to the
    client; we handle that case explicitly.  ``None`` means interactive
    plotting (used when running the scripts outside the web server).
    """

    # ``BytesIO`` is imported lazily so we don't add an actual dependency on
    # ``io`` when the package is used for file-based generation only.
    from io import BytesIO

    if save_path:
        if isinstance(save_path, BytesIO):
            fig.savefig(save_path, bbox_inches="tight")
            # don't close the figure until after the buffer is read by caller
        else:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
    else:  # interactive
        fig.show()


# ---- public plotting functions -------------------------------------------

def plot_learning_curve(save_path: Optional[str] = None) -> None:
    """Plot the (smoothed) episode rewards logged during training."""
    arr_path = os.path.join(LOG_DIR, "episode_rewards.npy")
    if not os.path.exists(arr_path):
        print("[dashboard] no episode_rewards.npy file found")
        return
    rewards = np.load(arr_path)
    if rewards.size == 0:
        print("[dashboard] episode_rewards.npy is empty")
        return

    window = 500
    if rewards.size < window:
        smooth = rewards
    else:
        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(smooth, color="blue")
    ax.set_title("AI Boss Reward Progress (Smoothed)")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")
    ax.grid(True)
    _maybe_save_or_show(fig, save_path or os.path.join(DEFAULT_VIS_DIR, "learning_curve.png"))


def plot_action_distribution(save_path: Optional[str] = None) -> None:
    """Histogram showing how often each action was chosen."""
    log = _load_log_data()
    if not log:
        print("[dashboard] no decision log available for action distribution")
        return
    actions = [step.get("action") for step in log if "action" in step]
    if not actions:
        print("[dashboard] decision log contains no actions")
        return

    counts = np.bincount(actions)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(counts)), counts, color="purple", alpha=0.7)
    ax.set_title("Action Distribution")
    ax.set_xlabel("Action index")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y")
    _maybe_save_or_show(fig, save_path or os.path.join(DEFAULT_VIS_DIR, "action_distribution.png"))


def plot_policy_confidence(save_path: Optional[str] = None) -> None:
    """Smoothed maximum Q‑value over all decisions (a rough "confidence")."""
    log = _load_log_data()
    if not log:
        print("[dashboard] no decision log available for policy confidence")
        return
    max_q = [max(step.get("q_values", [0])) for step in log]
    if not max_q:
        print("[dashboard] decision log contains no q_values")
        return

    window = 1000
    if len(max_q) < window:
        smooth = np.array(max_q)
    else:
        smooth = np.convolve(max_q, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(smooth, color="green")
    ax.set_title("Policy Confidence (smoothed max Q)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Max Q value")
    ax.grid(True)
    _maybe_save_or_show(fig, save_path or os.path.join(DEFAULT_VIS_DIR, "policy_confidence.png"))


def plot_decision_explanation(save_path: Optional[str] = None, index: int = 0) -> None:
    """Show Q‑values for a single logged decision (default first)."""
    log = _load_log_data()
    if not log or index >= len(log):
        print("[dashboard] cannot plot decision explanation – log is empty or index out of range")
        return
    q_vals = log[index].get("q_values")
    if not q_vals:
        print("[dashboard] selected log entry has no q_values")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(q_vals)), q_vals, color="orange", alpha=0.8)
    ax.set_title(f"Q-values for decision #{index}")
    ax.set_xlabel("Action index")
    ax.set_ylabel("Q value")
    ax.grid(axis="y")
    _maybe_save_or_show(fig, save_path or os.path.join(DEFAULT_VIS_DIR, "decision_explanation.png"))


def plot_state_space_tsne(save_path: Optional[str] = None) -> None:
    """Embed logged states with t-SNE and colour by chosen action."""
    if TSNE is None:
        print("[dashboard] sklearn required for t-SNE chart")
        return
    log = _load_log_data()
    states = [step.get("state") for step in log if "state" in step]
    acts = [step.get("action") for step in log if "action" in step]
    if not states:
        print("[dashboard] decision log contains no states for t-SNE")
        return

    X = np.array(states)
    tsne = TSNE(n_components=2, random_state=42)
    emb = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=acts, cmap="tab10", s=5, alpha=0.6)
    ax.set_title("State-space t-SNE (coloured by action)")
    ax.set_xlabel("TSNE 1")
    ax.set_ylabel("TSNE 2")
    _maybe_save_or_show(fig, save_path or os.path.join(DEFAULT_VIS_DIR, "state_tsne.png"))


def plot_reward_comparison(save_path: Optional[str] = None) -> None:
    """Show reward comparison when both traditional and ai logs exist.

    The traditional array is read from ``logs/traditional_rewards.npy``; it may
    not exist if evaluation wasn’t run.  When only the AI series is available a
    simple single-line chart is drawn with a notice printed to stderr.
    """
    ai_path = os.path.join(LOG_DIR, "episode_rewards.npy")
    trad_path = os.path.join(LOG_DIR, "traditional_rewards.npy")

    if not os.path.exists(ai_path):
        print("[dashboard] no ai rewards file found for comparison")
        return

    ai = np.load(ai_path)
    trad = None
    if os.path.exists(trad_path):
        trad = np.load(trad_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    if trad is not None:
        ax.plot(np.convolve(trad, np.ones(5)/5, mode="valid"),
                label="Traditional Boss", color="red", alpha=0.6)
    else:
        print("[dashboard] traditional reward data missing; plotting AI only")
    ax.plot(np.convolve(ai, np.ones(5)/5, mode="valid"),
            label="AI-Trained Boss", color="green", alpha=0.8)

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Total Reward")
    ax.set_title("Reward Comparison: Traditional vs AI Boss")
    ax.legend()
    ax.grid(True)
    _maybe_save_or_show(fig, save_path or os.path.join(DEFAULT_VIS_DIR, "reward_comparison.png"))


def generate_all(save_dir: Optional[str] = None) -> None:
    """Regenerate every chart and save into ``save_dir`` (defaults
to the visualizations folder).
    """
    dest = save_dir or DEFAULT_VIS_DIR
    os.makedirs(dest, exist_ok=True)
    plot_learning_curve(os.path.join(dest, "learning_curve.png"))
    plot_reward_comparison(os.path.join(dest, "reward_comparison.png"))
    plot_action_distribution(os.path.join(dest, "action_distribution.png"))
    plot_policy_confidence(os.path.join(dest, "policy_confidence.png"))
    plot_decision_explanation(os.path.join(dest, "decision_explanation.png"))
    plot_state_space_tsne(os.path.join(dest, "state_tsne.png"))


# re-export helper used by main.py
# ``visualize_rewards`` lives at the repository top level, so import
# it normally.  Using an absolute import avoids issues when the package is
# loaded as a namespace package (e.g. running the dashboard script directly).
try:
    from visualize_rewards import visualize_rewards
except ImportError:  # fallback if someone runs from a different cwd
    from .visualize_rewards import visualize_rewards  # type: ignore  # pragma: no cover

__all__ = [
    "generate_all",
    "plot_learning_curve",
    "plot_reward_comparison",
    "plot_action_distribution",
    "plot_policy_confidence",
    "plot_decision_explanation",
    "plot_state_space_tsne",
    "visualize_rewards",
]
