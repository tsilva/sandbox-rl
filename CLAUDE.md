# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **learning/testing sandbox** for reinforcement learning on Gymnasium environments.

### Typical Workflow

When a user asks to **"solve"** an environment:

1. **If no script exists**, write a single self-contained file for the env/algorithm combo (e.g., `ppo_cartpole.py`)
2. **Check the Gymnasium docs** to find the reward threshold that defines "solved" for that environment
3. **Run training** — each run creates a session with detailed logs
4. **Analyze results** and iterate with adjusted hyperparameters based on diagnosis
5. **Once solved**, run with `--play` to demonstrate the trained policy

**Solving Objective**: Achieve maximum reward in the **minimum number of timesteps**. Each run should learn from previous attempts.

**Session Structure**:
- If user says "try N times" → do exactly N runs, improving each time
- If no count specified → do as many runs as needed to hit the reward threshold

The session log (`session.log`) documents the reasoning behind each run, making it educational for learning RL engineering thinking.

The goal is educational — each file serves as a readable, runnable reference implementation that can be studied, modified, and learned from.

## Design Philosophy

**One flat file per environment/algorithm combination.** Each script (e.g., `ppo_cartpole.py`, `ppo_mountaincar.py`) must be completely self-contained:

- **No imports from other project files** — only external libraries (torch, numpy, gymnasium, etc.)
- **Duplicate code freely** — copy/paste shared logic between files rather than abstracting
- **No shared modules, utilities, or base classes** — every file stands alone
- **Easy to read, copy, and modify independently** — each file is a complete implementation

This intentional duplication keeps each implementation simple, portable, and easy to understand without jumping between files.

## Commands

```bash
# Install dependencies
pip install torch numpy gymnasium

# Train (creates new session with logging)
python ppo_cartpole.py
python ppo_cartpole.py --reason "Testing higher learning rate" --hp lr=5e-4

# Continue existing session with new run
python ppo_cartpole.py --session sessions/cartpole_20260117_100000 \
    --reason "Increasing entropy for exploration" \
    --diagnosis "Previous run plateaued early" \
    --hp entropy_coef=0.02

# Available hyperparameter overrides (--hp key=value)
#   n_envs, n_steps, batch_size, n_epochs, lr, gamma, lam,
#   clip_eps, value_coef, entropy_coef, max_iterations

# Play with trained models
python ppo_cartpole.py --play              # Watch trained agent
python ppo_cartpole.py --play --episodes 10  # Specify episode count
```

## Session Structure

Each training run creates a session directory:
```
sessions/cartpole_20260117_100000/
├── session.log          # High-level log: reasoning, params, diagnosis per run
├── best_model.pt        # Best model from successful solve
├── run_001/
│   ├── config.json      # Hyperparameters used
│   ├── run.log          # Detailed training output
│   ├── metrics.json     # Per-iteration metrics
│   └── checkpoint_best.pt
└── run_002/
    └── ...
```

## Architecture

Each PPO implementation (`ppo_*.py`) follows the same structure:

- **ActorCritic**: Neural network with separate actor (policy) and critic (value) heads, both using 64-64 hidden layers with Tanh activation
- **RolloutBuffer**: Stores trajectory data for vectorized environments
- **compute_gae_vectorized**: Computes Generalized Advantage Estimation
- **ppo_update**: Performs clipped PPO policy updates
- **train()**: Main training loop using `gym.vector.SyncVectorEnv` for parallel data collection
- **play()**: Loads saved model and renders episodes

Models are saved to `models/ppo_<env>.pt` with checkpoint format containing `model_state_dict`, `obs_dim`, and `act_dim`.

## MountainCar-Specific

MountainCar uses reward shaping (`shape_reward_vectorized`) to address sparse rewards:
- Height reward based on position
- Velocity reward to encourage momentum building
- Goal bonus for reaching the flag
