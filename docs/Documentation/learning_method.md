# Learning Method: DQN Family (No MCTS)

## Overview

The training system uses **pure Deep Q-Network (DQN)** variants — specifically **Double DQN** — with **no Monte Carlo Tree Search (MCTS)** whatsoever. There is no tree search of any kind during training. The agent learns entirely from self-generated experience stored in a replay buffer, bootstrapping from its own Q-value estimates.

All MCTS-like terms in the codebase (`student_agents/`) are post-training inference-only agents that wrap the trained DQN with minimax search at test time. They play no role in training.

## Algorithm: Double DQN (DDQN)

### Value Propagation: TD(0) Bootstrapping

The agent uses **Temporal-Difference (TD) learning**, not Monte Carlo returns. For each transition `(s, a, r, s', done)`:

```
target = r + γ · (1 − done) · Q_target(s', a_best)
```

Where:
- `a_best` is chosen by `Q_online(s')` (Double DQN: action decoupling)
- `Q_target(s', a_best)` evaluates that action using the target network
- Only **legal actions** are considered (argmax masked by legal mask)
- Terminal states have `max_next_q = 0`

This is a pure TD(0) bootstrap — the target value for `Q(s, a)` is computed from the immediate reward `r` plus the estimated value of the next state. No full-episode return is ever computed.

### Reward Design: Fully Sparse

The reward is **entirely sparse** — non-zero only at terminal states:

| Outcome | Reward |
|---------|--------|
| Agent wins | `+1.0` |
| Draw | `0.0` |
| Agent loses | `-1.0` |
| Illegal move | `-1.0` (immediate terminal) |

All non-terminal transitions store `reward = 0.0`. This means the Q-function must learn to propagate the terminal outcome backward through long sequences of zero-reward steps via TD bootstrapping with `γ = 0.99`.

### Target Network: Polyak Soft-Update

- **Default mode**: soft (Polyak) update every train step with `τ = 0.005`
  ```
  θ_target ← τ · θ_online + (1 − τ) · θ_target
  ```
- **Fallback mode** (`τ = 0.0`): hard copy of `θ_online` → `θ_target` every `target_update_freq = 500` steps

## Network Architecture

Small convolutional network with two spatial layers:

```
Input:  (B, 1, N, N)          — board relative to current player (+1, -1, 0)
  ↓
Conv2D(1→64, 3×3, pad=1) + ReLU
  ↓
Conv2D(64→64, 3×3, pad=1) + ReLU
  ↓
Flatten → Linear(64·N² → 128) + ReLU
  ↓
Linear(128 → N²+1)            — Q-values per cell + pass action
```

- Output dimension: `N² + 1` (all board cells + the pass action)
- Loss: **Smooth L1 (Huber)**, computed per-sample (required for PER weighting)
- Optimizer: **Adam**, `lr = 1e-3`
- Gradient clipping: `max_norm = 10.0`

## Exploration: ε-Greedy

Standard ε-greedy exploration with exponential decay:

```
ε ← max(ε_end, ε · ε_decay)
```

Defaults: `ε_start = 1.0`, `ε_end = 0.05`, `ε_decay = 0.9995`.

Additionally, each episode begins with `[1, max_random_training_plies]` random plies per player (default max = 4) to improve state-space coverage.

## Guided DQN (Optional Heuristic Exploration)

When `heuristic_weight > 0`, action selection becomes:

```
a = argmax[ Q(s, a) + hw · H(s, a) ]
```

The heuristic `H(s, a)` is a handcrafted positional score:

| Board Position | Score |
|---------------|-------|
| Corner | +5.0 |
| Edge (non-corner) | +1.0 |
| C-square (adjacent to empty corner) | −4.0 |
| Pass | −1.0 |
| Other | 0.0 |

**This is NOT reward shaping** — the heuristic only biases which actions are taken/explored; the stored reward is always the true sparse outcome. The heuristic weight is linearly annealed over episodes (default: from `0.0` to `0.0`, so effectively off unless set).

## Prioritized Experience Replay (PER — Optional)

When `use_per = True`, the replay buffer uses **proportional prioritization** (Schaul et al., 2016):

- Priority: `p = |TD-error| + ε`
- Sampling probability: `P(i) = pᵢ^α / Σ pⱼ^α` (α = 0.6)
- Importance sampling weight: `wᵢ = (N · P(i))^−β`, normalized by max weight
- β anneals linearly: `β_start = 0.4` → `1.0` over `per_beta_frames = 100,000` steps

## Self-Play Option (Separate Script)

A `train_vs_selfplay.py` variant trains the DQN against **frozen snapshots of itself**, updated every 500 episodes. This uses the same DQN algorithm — no MCTS.

## Opponent Curriculum (`train_vs_minimax.py`)

The agent trains against a curriculum of opponents of increasing difficulty:

| Stage | Progress Range | Opponents |
|-------|---------------|-----------|
| 1 — Early | `< 25%` | Random, Greedy, Heuristic |
| 2 — Mid | `25%–50%` | + FastMinimax (small weight ~39%) |
| 3 — Late | `50%–75%` | + FastMinimax (increasing to ~69%) |
| 4 — Final | `> 75%` | FastMinimax dominates (70%+) |

Opponent selection is stochastic within each stage using weighted sampling. The **FastMinimaxAgent** is a C++ bitboard minimax with alpha-beta pruning (not MCTS). It is used as a black-box opponent — the DQN learns to exploit its weaknesses via pure Q-learning.

## Comparison: DQN vs MCTS

| Aspect | This Project (DQN) | Typical MCTS + NN |
|--------|-------------------|-------------------|
| **Search at training time** | None | Look-ahead tree search per move |
| **Value target** | TD(0) bootstrap: `r + γQ(s')` | Monte Carlo: full simulation returns |
| **Policy improvement** | ε-greedy over Q-values | Boltzmann over visit counts |
| **Sample efficiency** | Replay buffer reuses experience | Each simulation is on-policy |
| **Test-time search** | None (pure feed-forward) | MCTS tree search (AlphaZero style) |

The project's approach is simpler, faster, and more sample-efficient per episode, but does not benefit from look-ahead search during training or inference.

## Summary

- **Algorithm**: Double DQN (DDQN)
- **Value propagation**: TD(0) bootstrapping
- **MCTS used?**: No — zero tree search during training
- **Reward**: Sparse (+1/0/−1 at game end)
- **Network**: 2-layer ConvNet → 2-layer MLP
- **Replay**: Uniform or Prioritized (PER)
- **Target network**: Polyak soft-update (τ = 0.005)
- **Exploration bonus**: Corner/edge heuristic (Guided DQN, optional)
- **Opponents**: Curriculum of Random → Greedy → Heuristic → Minimax (C++ bitboard)
