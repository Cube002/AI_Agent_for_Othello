import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.join(BASE_DIR, "Training")

OPPONENT_COLORS = {
    "random": "#2E86AB",
    "greedy": "#A23B72",
    "heuristic": "#F18F01",
    "fast_minimax": "#C73E1D",
}

OPPONENT_LABELS = {
    "random": "Random",
    "greedy": "Greedy",
    "heuristic": "Heuristic",
    "fast_minimax": "Minimax d3",
}

METRIC_CONFIG = {
    "avg_reward": {"ylabel": "Average Reward", "title": "Average Reward vs Episode"},
    "loss": {"ylabel": "Loss", "title": "Loss vs Episode"},
    "mean_q": {"ylabel": "Mean Q", "title": "Mean Q vs Episode"},
    "grad_norm": {"ylabel": "Gradient Norm", "title": "Gradient Norm vs Episode"},
    "mean_td_error": {"ylabel": "Mean TD Error", "title": "Mean TD Error vs Episode"},
}

EXP_LINESTYLES = [
    ("o-", {"color": "#2E86AB", "markersize": 3}),
    ("s-", {"color": "#A23B72", "markersize": 3}),
    ("D-", {"color": "#F18F01", "markersize": 3}),
    ("^-", {"color": "#C73E1D", "markersize": 3}),
    ("v-", {"color": "#4CAF50", "markersize": 3}),
    ("<-", {"color": "#9C27B0", "markersize": 3}),
]
