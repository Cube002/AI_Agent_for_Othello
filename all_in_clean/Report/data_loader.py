import os
import glob
import re
import json
import pandas as pd

from .config import TRAINING_DIR


def load_experiment(exp_num):
    exp_num = str(exp_num).zfill(3)
    exp_dir = os.path.join(TRAINING_DIR, f"experiments_{exp_num}")
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    setup_path = os.path.join(exp_dir, "setup.json")
    with open(setup_path) as f:
        setup = json.load(f)

    results_dir = os.path.join(exp_dir, "results")
    train_path = os.path.join(results_dir, "latest_train.csv")
    train_df = pd.read_csv(train_path) if os.path.isfile(train_path) else pd.DataFrame()

    eval_files = sorted(
        glob.glob(os.path.join(results_dir, "latest_eval_ep*.csv")),
        key=lambda p: int(
            re.search(r"latest_eval_ep(\d+)\.csv", os.path.basename(p)).group(1)
        ),
    )
    eval_rows = []
    for f in eval_files:
        ep = int(
            re.search(r"latest_eval_ep(\d+)\.csv", os.path.basename(f)).group(1)
        )
        df = pd.read_csv(f)
        df["eval_episode"] = ep
        eval_rows.append(df)
    eval_df = pd.concat(eval_rows, ignore_index=True) if eval_rows else pd.DataFrame()

    return {
        "exp_num": exp_num,
        "setup": setup,
        "train_df": train_df,
        "eval_df": eval_df,
    }


def get_opponents(eval_df):
    if eval_df.empty:
        return []
    return sorted(
        eval_df[eval_df["role"] == "aggregate"]["opponent"].unique()
    )


def get_train_metrics(train_df):
    if train_df.empty:
        return []
    base = {"episode", "epsilon", "beta"}
    numeric = set(train_df.select_dtypes(include="number").columns)
    return sorted(numeric - base)


def label_from_exp(exp_num, setup=None):
    return f"exp_{exp_num}"
