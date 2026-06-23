"""
eval_overnight.py — Evaluate all overnight training checkpoints
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from evaluation import evaluate_fair
from agent import DQNAgent
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.heuristic_agent import HeuristicAgent
from agents.minimax_agent import MinimaxAgent

OPPONENTS = {"random": RandomAgent, "greedy": GreedyAgent,
             "heuristic": HeuristicAgent, "minimax": MinimaxAgent}
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

ckpts = sorted(glob.glob(os.path.join(MODELS_DIR, "guided_per_dqn_6_overnight_ep*.pth")))
ckpts.append(os.path.join(MODELS_DIR, "guided_per_dqn_6_overnight_final.pth"))

print(f"{'Checkpoint':<45} {'random':<8} {'greedy':<8} {'heuristic':<10} {'minimax':<9} {'avg':<6}")
print("-" * 86)

for ckpt in ckpts:
    if not os.path.exists(ckpt):
        continue
    try:
        agent = DQNAgent(board_size=6, use_per=True, heuristic_weight=0.2)
        agent.load(ckpt)
        agent.q_net.eval()
        scores = {}
        for opp_name, opp_cls in OPPONENTS.items():
            res = evaluate_fair(agent, opp_cls, board_size=6, n_games=100, random_opening_plies=2)
            scores[opp_name] = res["score"]
        avg = np.mean(list(scores.values()))
        name = os.path.basename(ckpt)
        print(f"{name:<45} {scores['random']:<8.3f} {scores['greedy']:<8.3f} "
              f"{scores['heuristic']:<10.3f} {scores['minimax']:<9.3f} {avg:<6.3f}")
    except Exception as e:
        print(f"{os.path.basename(ckpt):<45} ERROR: {e}")

# Also evaluate the original best model for comparison
print("\nComparison with original:")
orig = os.path.join(MODELS_DIR, "guided_per_dqn_6.pth")
agent = DQNAgent(board_size=6, use_per=True, heuristic_weight=0.2)
agent.load(orig)
agent.q_net.eval()
scores = {}
for opp_name, opp_cls in OPPONENTS.items():
    res = evaluate_fair(agent, opp_cls, board_size=6, n_games=100)
    scores[opp_name] = res["score"]
avg = np.mean(list(scores.values()))
print(f"{'guided_per_dqn_6.pth (original)':<45} {scores['random']:<8.3f} {scores['greedy']:<8.3f} "
      f"{scores['heuristic']:<10.3f} {scores['minimax']:<9.3f} {avg:<6.3f}")
