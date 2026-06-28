import os
import matplotlib.pyplot as plt

from .config import EXP_LINESTYLES
from .smoothing import rolling_smooth
from .data_loader import label_from_exp


def plot_td_error(experiments, output_dir, smooth=0):
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, exp in enumerate(experiments):
        df = exp["train_df"]
        if df.empty or "mean_td_error" not in df.columns:
            continue
        label = label_from_exp(exp["exp_num"], exp.get("setup"))
        y = rolling_smooth(df["mean_td_error"], smooth) if smooth > 0 else df["mean_td_error"]
        style = EXP_LINESTYLES[idx % len(EXP_LINESTYLES)]
        ax.plot(df["episode"], y, style[0], label=label, **style[1])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean TD Error")
    ax.set_title("Mean TD Error vs Episode")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "mean_td_error.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
