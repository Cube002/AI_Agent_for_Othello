# Experiment Comparison — Training Parameters

Values in **bold** differ from the default across the six experiments.

| Parameter | 007 | 008 | 026 | 027 | 028 | 029 |
|---|---|---|---|---|---|---|
| **learning_rate** | 0.0005 | 0.0005 | **5e-5** | **0.005** | 0.0005 | 0.0005 |
| **epsilon_start** | 0.05 | 0.05 | 0.05 | 0.05 | **0.1** | 0.05 |
| **epsilon_end** | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| **epsilon_decay** | 0.9995 | 0.9995 | 0.9995 | 0.9995 | 0.9995 | 0.9995 |
| **minimax_start_progress** | 0.05 | **0.01** | 0.05 | 0.05 | 0.05 | 0.05 |
| **minimax_full_progress** | 0.3 | **0.1** | 0.3 | 0.3 | 0.3 | 0.3 |
| **heuristic_weight** | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | **0.3** |
| **double_dqn** | true | true | true | true | true | true |
| **use_per** | true | true | true | true | true | true |
| **per_alpha** | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 |
| **per_beta_start** | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 |
| **per_beta_frames** | 100k | 100k | 100k | 100k | 100k | 100k |
| **batch_size** | 128 | 128 | 128 | 128 | 128 | 128 |
| **buffer_capacity** | 50k | 50k | 50k | 50k | 50k | 50k |
| **target_update_freq** | 500 | 500 | 500 | 500 | 500 | 500 |
| **learning_starts** | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| **num_episodes** | 200k | 200k | 200k | 200k | 200k | 200k |
| **seed** | 42 | 42 | 42 | 42 | 42 | 42 |

### Identical Parameters (not shown above)

| Parameter | Value |
|---|---|
| minimax_max_depth | 3 |
| minimax_time_limit | 1.0 s |
| final_minimax_weight | 0.6 |
| gamma | 0.99 |
| tau | 0.005 |
| max_random_training_plies | 4 |
| print_every | 50 |
| eval_every | 1000 |
| save_every | 1000 |

### Summary of Changes

| Experiment | What changed vs. baseline (007/008) |
|---|---|
| **007** | Baseline — first run with PER + curriculum |
| **008** | Minimax curriculum starts **earlier** (start_progress=0.01, full_progress=0.1) |
| **026** | **Lower learning rate** (5e-5 vs 5e-4) |
| **027** | **Higher learning rate** (5e-3 vs 5e-4) |
| **028** | **Higher initial epsilon** (0.1 vs 0.05) — more exploration |
| **029** | **Higher heuristic_weight** (0.3 vs 0.2) |
