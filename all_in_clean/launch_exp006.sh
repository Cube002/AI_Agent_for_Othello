#!/bin/bash
set -e
cd /home/cube/projects/AI_Agent_for_Othello/all_in_clean
rm -rf Training/experiments_006
tmux kill-session -t othello_exp006 2>/dev/null || true
tmux new-session -d -s othello_exp006 "cd /home/cube/projects/AI_Agent_for_Othello/all_in_clean && /home/cube/.local/bin/uv run python advanced_training.py --base_dir Training/experiments --load_model_path models/fork_experiment002_latest.pth --batch_size 64 2>&1"
echo "launched"
