import os
import matplotlib.pyplot as plt

from .config import EXP_LINESTYLES
from .smoothing import rolling_smooth
from .data_loader import label_from_exp


def plot_grad_norm(experiments, output_dir, smooth=0):
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, exp in enumerate(experiments):
        df = exp["train_df"]
        if df.empty or "grad_norm" not in df.columns:
            continue
        label = label_from_exp(exp["exp_num"], exp.get("setup"))
        y = rolling_smooth(df["grad_norm"], smooth) if smooth > 0 else df["grad_norm"]
        style = EXP_LINESTYLES[idx % len(EXP_LINESTYLES)]
        ax.plot(df["episode"], y, style[0], label=label, **style[1])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm vs Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "grad_norm.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
