import numpy as np


def compute_binned_winrate(eval_df, win_step=10000, win_size=10000):
    """
    Bin evaluation data by episode.

    For each bin centered at multiples of *win_step*,
    collect all evaluation points within [center - win_size/2, center + win_size/2)
    and compute mean + std of the `score` column.

    Returns dict: opponent_name -> [(bin_center, mean, std)]
    """
    if eval_df.empty:
        return {}

    eval_df = eval_df[eval_df["role"] == "aggregate"].copy()
    if eval_df.empty:
        return {}

    results = {}
    half = win_size // 2
    max_ep = eval_df["eval_episode"].max()

    for opponent in sorted(eval_df["opponent"].unique()):
        opp_df = eval_df[eval_df["opponent"] == opponent]
        bins = []
        for center in range(0, max_ep + win_step, win_step):
            lo = center - half
            hi = center + half
            mask = (opp_df["eval_episode"] >= lo) & (opp_df["eval_episode"] < hi)
            subset = opp_df.loc[mask, "score"]
            if len(subset) > 0:
                bins.append((center, subset.mean(), subset.std()))
        results[opponent] = bins

    return results


def rolling_smooth(series, window=10):
    """Apply a centered rolling mean with given window size."""
    if window < 2 or len(series) < window:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()
