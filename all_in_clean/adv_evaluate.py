"""
adv_evaluate.py — Evaluate checkpoints or plot progress from existing CSV results.

Modes:
  (default)  Plot winrate from existing latest_eval_ep*.csv files.
  --model    Evaluate a single .pth checkpoint.
  --model-list  Evaluate a comma/space-separated list of .pth files.
  --models-dir  Evaluate all model_ep*.pth checkpoints in a directory.
"""

import sys, os, glob, re
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
import argparse
from evaluation import evaluate_fair


DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "Training", "experiments_017", "results",
)
DEFAULT_MODELS_DIR = os.path.join(
    os.path.dirname(__file__),
    "Training", "experiments_01", "model_checkpoints",
)


def plot_from_csv(results_dir):
    csv_files = sorted(
        glob.glob(os.path.join(results_dir, "latest_eval_ep*.csv")),
        key=lambda p: int(
            re.search(r"latest_eval_ep(\d+)\.csv", os.path.basename(p)).group(1)
        ),
    )

    if not csv_files:
        print(f"No latest_eval_ep*.csv files found in {results_dir}")
        return

    episodes = []
    data = {}

    for csv_file in csv_files:
        ep = int(
            re.search(r"latest_eval_ep(\d+)\.csv", os.path.basename(csv_file)).group(1)
        )
        episodes.append(ep)

        with open(csv_file) as f:
            f.readline()
            for line in f:
                parts = line.strip().split(",")
                opp, role, wins, draws, losses, wr, score = parts
                key = opp if role == "aggregate" else f"{opp}_{role}"
                data.setdefault(key, []).append(float(score))

    header = f"{'Episode':<10}" + "".join(f"{k:<14s}" for k in data)
    print("\n" + "=" * (10 + 14 * len(data)))
    print(header)
    print("-" * (10 + 14 * len(data)))
    for i, ep in enumerate(episodes):
        row = f"{ep:<10d}" + "".join(f"{data[k][i]:<14.3f}" for k in data)
        print(row)
    print("=" * (10 + 14 * len(data)))

    plt.figure(figsize=(10, 6))
    for key, scores in data.items():
        plt.plot(episodes, scores, marker="o", label=key)
    plt.xlabel("Training Episodes")
    plt.ylabel("Score (win_rate)")
    plt.title(f"Win Rate Progress — {os.path.basename(os.path.dirname(results_dir))}")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(results_dir, "..", "eval_winrate_vs_episodes.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {out_path}")


def run_eval_single(ckpt, label, opponents, n_games, board_size, random_opening_plies, device):
    from agent import DQNAgent
    agent = DQNAgent(board_size=board_size, use_per=True, heuristic_weight=0.2, device=device)
    agent.load(ckpt)
    agent.q_net.eval()

    results = {}
    print(f"\n--- {label} ({os.path.basename(ckpt)}) ---")
    for opp_name, opp_fn in opponents.items():
        res = evaluate_fair(
            agent, opp_fn,
            board_size=board_size,
            n_games=n_games,
            random_opening_plies=random_opening_plies,
        )
        score = res["score"]
        results[opp_name] = score
        print(f"  vs {opp_name:<14s}  score={score:.3f}  "
              f"({res['wins']}W/{res['draws']}D/{res['losses']}L)")
    return results


def run_eval(models_dir=None, model=None, model_list=None,
             n_games=100, depth=5, board_size=6, random_opening_plies=2,
             opponents_filter=None, device=None,
             output=None, no_plot=False):
    from evaluation import evaluate_fair
    from agent import DQNAgent
    from agents.random_agent import RandomAgent
    from agents.heuristic_agent import HeuristicAgent
    from agents.cpp_minimax_agent import FastMinimaxAgent

    all_opponents = {
        "random": RandomAgent,
        "heuristic": HeuristicAgent,
        f"minimax_d{depth}": lambda bs: FastMinimaxAgent(board_size=bs, max_depth=depth),
    }

    if opponents_filter:
        opponents = {k: v for k, v in all_opponents.items() if k in opponents_filter}
        missing = set(opponents_filter) - set(opponents)
        if missing:
            print(f"Warning: unknown opponents ignored: {missing}")
    else:
        opponents = all_opponents

    # Collect checkpoints and their labels
    ckpt_info = []  # (path, label)

    if model is not None:
        if not os.path.isfile(model):
            print(f"Error: model file not found: {model}")
            return
        label = os.path.splitext(os.path.basename(model))[0]
        ckpt_info.append((model, label))

    if model_list is not None:
        for p in model_list:
            p = p.strip()
            if not os.path.isfile(p):
                print(f"Warning: model file not found, skipping: {p}")
                continue
            label = os.path.splitext(os.path.basename(p))[0]
            ckpt_info.append((p, label))

    if models_dir is not None:
        found = sorted(
            glob.glob(os.path.join(models_dir, "model_ep*.pth")),
            key=lambda p: int(
                re.search(r"model_ep(\d+)\.pth", os.path.basename(p)).group(1)
            ),
        )
        if not found:
            print(f"No model_ep*.pth files found in {models_dir}")
        for ckpt in found:
            ep = int(re.search(r"model_ep(\d+)\.pth", os.path.basename(ckpt)).group(1))
            ckpt_info.append((ckpt, f"Episode {ep}"))

    if not ckpt_info:
        print("No checkpoints to evaluate. Use --model, --model-list, or --models-dir.")
        return

    labels = []
    all_results = {name: [] for name in opponents}

    for ckpt, label in ckpt_info:
        labels.append(label)
        res = run_eval_single(ckpt, label, opponents,
                              n_games, board_size, random_opening_plies, device)
        for opp_name in opponents:
            all_results[opp_name].append(res[opp_name])

    # Print summary table
    header = f"{'Model':<20}" + "".join(f"{name:<14s}" for name in opponents)
    print("\n" + "=" * (20 + 14 * len(opponents)))
    print(header)
    print("-" * (20 + 14 * len(opponents)))
    for i, lab in enumerate(labels):
        row = f"{lab:<20}" + "".join(f"{all_results[name][i]:<14.3f}" for name in opponents)
        print(row)
    print("=" * (20 + 14 * len(opponents)))

    if no_plot:
        return

    plt.figure(figsize=(10, 6))
    for opp_name in opponents:
        plt.plot(range(len(labels)), all_results[opp_name], marker="o", label=opp_name)
    plt.xlabel("Checkpoint")
    plt.ylabel("Score (win_rate)")
    src = model or (model_list[0] if model_list else models_dir)
    title_base = os.path.basename(os.path.dirname(src)) if src and os.path.isdir(os.path.dirname(src)) else "eval"
    plt.title(f"Model Win Rate vs Opponents — {title_base}")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.legend()
    plt.grid(True)

    if output is None:
        if models_dir:
            output = os.path.join(models_dir, "..", "eval_winrate_vs_episodes.png")
        else:
            output = os.path.join(os.path.dirname(__file__), "eval_winrate.png")
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot evaluation progress from CSV results, or re-evaluate models."
    )

    # Plot mode
    parser.add_argument(
        "--results-dir", default=DEFAULT_RESULTS_DIR,
        help="Directory with latest_eval_ep*.csv files (plot mode)",
    )

    # Eval mode — model sources (mutually exclusive-ish)
    parser.add_argument(
        "--model", default=None,
        help="Evaluate a single .pth checkpoint file",
    )
    parser.add_argument(
        "--model-list", default=None,
        help="Comma-separated list of .pth checkpoint files to evaluate",
    )
    parser.add_argument(
        "--models-dir", default=None,
        help="Directory with model_ep*.pth checkpoints (all evaluated)",
    )

    # Eval mode — hyper-parameters
    parser.add_argument("--n-games", type=int, default=100,
                        help="Number of games per opponent pair")
    parser.add_argument("--depth", type=int, default=5,
                        help="Minimax search depth")
    parser.add_argument("--board-size", type=int, default=6,
                        help="Othello board size (default: 6)")
    parser.add_argument("--random-opening-plies", type=int, default=2,
                        help="Random opening moves for game diversity")
    parser.add_argument("--device", default=None,
                        help="Device: 'cpu', 'cuda', or None for auto")

    # Eval mode — opponents filter
    parser.add_argument(
        "--opponents", default=None,
        help="Comma-separated list of opponents: random,heuristic,minimax_d<N>",
    )

    # Eval mode — output
    parser.add_argument(
        "--output", default=None,
        help="Output path for the plot image",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plotting, just print results",
    )

    args = parser.parse_args()

    # Determine mode: eval if any model source is given, else plot
    eval_mode = args.model is not None or args.model_list is not None or args.models_dir is not None

    if eval_mode:
        if args.model_list is not None:
            args.model_list = [p.strip() for p in args.model_list.split(",")]
        if args.opponents is not None:
            args.opponents = [o.strip() for o in args.opponents.split(",")]
        run_eval(
            models_dir=args.models_dir,
            model=args.model,
            model_list=args.model_list,
            n_games=args.n_games,
            depth=args.depth,
            board_size=args.board_size,
            random_opening_plies=args.random_opening_plies,
            opponents_filter=args.opponents,
            device=args.device,
            output=args.output,
            no_plot=args.no_plot,
        )
    else:
        plot_from_csv(args.results_dir)
