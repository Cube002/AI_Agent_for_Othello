import os
import matplotlib.pyplot as plt
import numpy as np

from .config import OPPONENT_COLORS, OPPONENT_LABELS, EXP_LINESTYLES
from .smoothing import compute_binned_winrate
from .data_loader import get_opponents, label_from_exp


def plot_winrate(experiments, output_dir, win_step=10000, win_size=10000):
    """Two sets of winrate plots:
    1. Per-opponent: one figure per opponent, all experiments overlaid.
    2. Per-experiment: one figure per experiment, all opponents overlaid.
    """
    # Precompute binned data for all experiments
    all_binned = {}
    for exp in experiments:
        label = label_from_exp(exp["exp_num"], exp.get("setup"))
        binned = compute_binned_winrate(exp["eval_df"], win_step, win_size)
        all_binned[label] = binned

    opponents = _detect_all_opponents(all_binned)

    # --- Per-opponent plots ---
    for opp in opponents:
        fig, ax = _figure()
        for idx, (label, binned) in enumerate(sorted(all_binned.items())):
            if opp not in binned or not binned[opp]:
                continue
            centers, means, stds = zip(*binned[opp])
            centers = np.array(centers)
            means = np.array(means)
            stds = np.array(stds)
            style = EXP_LINESTYLES[idx % len(EXP_LINESTYLES)]
            ax.plot(centers, means, style[0], label=label, **style[1])
            ax.fill_between(centers, means - stds, means + stds, alpha=0.15, color=style[1]["color"])
        _finish(fig, ax,
                xlabel="Episode",
                ylabel="Score (win_rate)",
                path=os.path.join(output_dir, f"winrate_vs_{opp}.pdf"))

    # --- Per-experiment plots ---
    for exp in experiments:
        label = label_from_exp(exp["exp_num"], exp.get("setup"))
        binned = all_binned.get(label, {})
        if not binned:
            continue
        fig, ax = _figure()
        for opp in opponents:
            if opp not in binned or not binned[opp]:
                continue
            centers, means, stds = zip(*binned[opp])
            centers = np.array(centers)
            means = np.array(means)
            stds = np.array(stds)
            color = OPPONENT_COLORS.get(opp, "#333333")
            ax.plot(centers, means, "o-", label=OPPONENT_LABELS.get(opp, opp),
                    color=color, markersize=3)
            ax.fill_between(centers, means - stds, means + stds,
                            alpha=0.15, color=color)
        _finish(fig, ax,
                xlabel="Episode",
                ylabel="Score (win_rate)",
                path=os.path.join(output_dir, f"{label}_winrate.pdf"))


def _detect_all_opponents(all_binned):
    seen = set()
    for binned in all_binned.values():
        seen.update(binned.keys())
    # Sort in a fixed order
    order = ["random", "greedy", "heuristic", "fast_minimax"]
    return [o for o in order if o in seen] + sorted(seen - set(order))


def _figure():
    fig, ax = plt.subplots(figsize=(5, 3))
    return fig, ax


def _k_formatter(x, _):
    return f"{int(x//1000)}k"


def _finish(fig, ax, xlabel, ylabel, path):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_k_formatter))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
