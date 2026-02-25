from flask import Flask, render_template, send_from_directory, redirect, url_for
import os
import numpy as np

# import the visualization helpers so that we can call them when needed
from visualizations import dashboard
from io import BytesIO
from flask import Response

app = Flask(
    __name__,
    static_folder="visualizations",
    template_folder="templates",
)


@app.route("/")
def index():
    """Home page listing any images that have been generated.

    Additionally compute a few simple statistics from the logs so the
    template can display them in the hero card.  Also provide a list of
    charts with human‑readable titles and descriptions so the page can
    be rendered dynamically rather than hard‑coding each graph in HTML.
    """
    images = []
    if os.path.isdir(app.static_folder):
        images = [f for f in os.listdir(app.static_folder) if f.lower().endswith(".png")]

    stats = {}
    try:
        rewards = np.load("logs/episode_rewards.npy")
        stats['episodes'] = len(rewards)
        stats['mean_reward'] = float(np.mean(rewards))
    except Exception:
        pass

    charts = [
        {
            'id': 'learning_curve',
            'title': 'Learning Curve',
            'desc': 'Smoothed training reward over episodes.',
        },
        {
            'id': 'reward_comparison',
            'title': 'Reward Comparison',
            'desc': 'Traditional vs AI boss rewards (if available).',
        },
        {
            'id': 'action_distribution',
            'title': 'Action Distribution',
            'desc': 'Frequency each boss action was selected during training.',
        },
        {
            'id': 'policy_confidence',
            'title': 'Policy Confidence',
            'desc': 'Smoothed maximum Q‑value over training steps.',
        },
        {
            'id': 'decision_explanation',
            'title': 'Decision Explanation',
            'desc': 'Q‑values for a representative decision (mid‑training).',
        },
        {
            'id': 'state_tsne',
            'title': 'State-space t‑SNE',
            'desc': 't‑SNE embedding of state vectors colored by chosen action.',
        },
    ]

    return render_template("index.html", images=images, stats=stats, charts=charts)


@app.route("/generate")
def generate():
    """Regenerate the static plots and then return to the index page."""
    # attempt to call the helper that saves to the visualizations folder
    try:
        dashboard.generate_all()
    except Exception as e:
        # log the failure; in a real app you might flash a message
        print(f"Error generating visuals: {e}")
    return redirect(url_for("index"))


def _fig_response(func, *args, **kwargs):
    buf = BytesIO()
    kwargs.setdefault('save_path', buf)
    func(*args, **kwargs)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype='image/png')


@app.route('/action_distribution.png')
def action_distribution_png():
    return _fig_response(dashboard.plot_action_distribution)


@app.route('/policy_confidence.png')
def policy_confidence_png():
    return _fig_response(dashboard.plot_policy_confidence)


@app.route('/decision_explanation.png')
def decision_explanation_png():
    # optional query param index?
    return _fig_response(dashboard.plot_decision_explanation)


@app.route('/learning_curve.png')
def learning_curve_png():
    return _fig_response(dashboard.plot_learning_curve)


@app.route('/reward_comparison.png')
def reward_comparison_png():
    return _fig_response(dashboard.plot_reward_comparison)


@app.route('/state_tsne.png')
def state_tsne_png():
    return _fig_response(dashboard.plot_state_space_tsne)


@app.route("/images/<path:filename>")
def images(filename):
    """Serve image assets directly from the visualizations folder."""
    return send_from_directory(app.static_folder, filename)


if __name__ == "__main__":
    # default port 5000, accessible on localhost
    app.run(debug=True)
