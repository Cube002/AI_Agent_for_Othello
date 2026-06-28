import os
import matplotlib.pyplot as plt

from .config import EXP_LINESTYLES
from .smoothing import rolling_smooth
from .data_loader import label_from_exp


def plot_avg_reward(experiments, output_dir, smooth=0):
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, exp in enumerate(experiments):
        df = exp["train_df"]
        if df.empty or "avg_reward" not in df.columns:
            continue
        label = label_from_exp(exp["exp_num"], exp.get("setup"))
        y = rolling_smooth(df["avg_reward"], smooth) if smooth > 0 else df["avg_reward"]
        style = EXP_LINESTYLES[idx % len(EXP_LINESTYLES)]
        ax.plot(df["episode"], y, style[0], label=label, **style[1])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Reward")
    ax.set_title("Average Reward vs Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "avg_reward.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
