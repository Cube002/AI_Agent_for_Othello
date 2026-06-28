"""
Orchestrate all Report plots.

Usage:
    uv run --no-sync python -m Report 007 008 026 027 028 029 [options]

Options:
    --smooth N      Rolling average window for train-metric plots (default: 0 = off)
    --win-step N    Step between winrate bin centers (default: 10000)
    --win-size N    Window size for winrate binning (default: 10000)
    --output-dir D  Output directory (default: Report/output)
"""

import sys
import os
import argparse

# Ensure the project root is on sys.path so imports from all_in_clean work
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from Report.data_loader import load_experiment
from Report.plot_winrate import plot_winrate
from Report.plot_avg_reward import plot_avg_reward
from Report.plot_loss import plot_loss
from Report.plot_td_error import plot_td_error
from Report.plot_mean_q import plot_mean_q
from Report.plot_grad_norm import plot_grad_norm


def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment report plots (PDF)."
    )
    parser.add_argument(
        "experiments", nargs="+",
        help="Experiment numbers, e.g. 007 008 026 027 028 029"
    )
    parser.add_argument(
        "--smooth", type=int, default=0,
        help="Rolling average window for training metric plots (default: 0 = off)"
    )
    parser.add_argument(
        "--win-step", type=int, default=10000,
        help="Step between winrate bin centers in episodes (default: 10000)"
    )
    parser.add_argument(
        "--win-size", type=int, default=10000,
        help="Window size for winrate binning in episodes (default: 10000)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: Report/output/exp_<list>)"
    )

    args = parser.parse_args()

    base_output = args.output_dir
    if base_output is None:
        exp_str = "_".join(sorted(args.experiments))
        base_output = os.path.join(
            os.path.dirname(__file__), "output", f"exp_{exp_str}"
        )
    os.makedirs(base_output, exist_ok=True)

    print(f"Loading experiments: {', '.join(args.experiments)}")
    experiments = []
    for exp_num in args.experiments:
        try:
            exp = load_experiment(exp_num)
            experiments.append(exp)
            setup = exp.get("setup", {})
            params = setup.get("training_params", {})
            print(f"  exp_{exp_num}: lr={params.get('learning_rate')}, "
                  f"eps_start={params.get('epsilon_start')}, "
                  f"double_dqn={params.get('double_dqn')}, "
                  f"use_per={params.get('use_per')}, "
                  f"heuristic_w={params.get('heuristic_weight')}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}", file=sys.stderr)

    if not experiments:
        print("No valid experiments found. Aborting.")
        sys.exit(1)

    print(f"\nGenerating winrate plots...")
    plot_winrate(experiments, base_output,
                 win_step=args.win_step, win_size=args.win_size)

    print(f"\nGenerating training metric plots (smooth={args.smooth})...")
    plot_avg_reward(experiments, base_output, smooth=args.smooth)
    plot_loss(experiments, base_output, smooth=args.smooth)
    plot_td_error(experiments, base_output, smooth=args.smooth)
    plot_mean_q(experiments, base_output, smooth=args.smooth)
    plot_grad_norm(experiments, base_output, smooth=args.smooth)

    print(f"\nAll plots saved to: {base_output}")


if __name__ == "__main__":
    main()
