<div align="center">
  <img src="logo.png" alt="sandbox-rl" width="512"/>

  # sandbox-rl

  [![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
  [![Gymnasium](https://img.shields.io/badge/Gymnasium-0081A5?logo=openaigym&logoColor=white)](https://gymnasium.farama.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

  **🎮 RL learning sandbox for solving Gymnasium environments 🧠**

  [Gymnasium Docs](https://gymnasium.farama.org/) · [PPO Paper](https://arxiv.org/abs/1707.06347)
</div>

---

## Overview

A sandbox for reinforcement learning experiments using Proximal Policy Optimization (PPO). Each implementation is a single, self-contained file that you can read, run, and modify independently.

**Why this project?**

- 📚 **Educational** — Each file is a complete, readable reference implementation
- 🔬 **Iterative** — Session logging tracks your reasoning across training runs
- 🧩 **Self-contained** — No shared modules; copy any file and it just works
- ⚡ **Vectorized** — Parallel environment execution for faster training

## Environments

| Environment | Script | Solve Threshold | Status |
|-------------|--------|-----------------|--------|
| CartPole-v1 | `ppo_cartpole.py` | ≥ 475 avg reward | ✅ Solved |
| MountainCar-v0 | `ppo_mountaincar.py` | ≥ -110 avg reward | ✅ Solved |
| LunarLander-v3 | `ppo_lunarlander.py` | ≥ 200 avg reward | ✅ Solved |
| CarRacing-v2 | `ppo_carracing.py` | ≥ 900 avg reward | 🚧 In Progress |

## Quick Start

```bash
# Clone and install
git clone https://github.com/tsilva/sandbox-rl.git
cd sandbox-rl
pip install torch numpy "gymnasium[box2d]"

# Train an agent
python ppo_cartpole.py

# Watch a trained agent
python ppo_cartpole.py --play
```

## Usage

### Training

```bash
# Basic training (creates new session)
python ppo_cartpole.py

# Train with custom hyperparameters
python ppo_cartpole.py --reason "Testing higher learning rate" --hp lr=5e-4

# Continue an existing session with a new run
python ppo_cartpole.py --session sessions/cartpole_20260117_100000 \
    --reason "Increasing entropy for exploration" \
    --diagnosis "Previous run plateaued early" \
    --hp entropy_coef=0.02
```

### Hyperparameter Overrides

Use `--hp key=value` to override any hyperparameter:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_envs` | 8 | Number of parallel environments |
| `n_steps` | 2048 | Steps per environment per rollout |
| `batch_size` | 64 | Minibatch size for updates |
| `n_epochs` | 10 | PPO epochs per rollout |
| `lr` | 3e-4 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `lam` | 0.95 | GAE lambda |
| `clip_eps` | 0.2 | PPO clipping epsilon |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |
| `value_coef` | 0.5 | Value loss coefficient |
| `max_iterations` | 500 | Maximum training iterations |

### Playing

```bash
python ppo_cartpole.py --play              # Watch trained agent
python ppo_cartpole.py --play --episodes 10  # Specify episode count
```

## Session Structure

Each training run creates a session directory with detailed logs for tracking experiments:

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

The session log documents your reasoning behind each run, making it useful for learning RL engineering thinking.

## Repository Structure

```
sandbox-rl/
├── ppo_cartpole.py        # PPO for CartPole-v1
├── ppo_mountaincar.py     # PPO for MountainCar-v0
├── ppo_lunarlander.py     # PPO for LunarLander-v3
├── ppo_carracing.py       # PPO for CarRacing-v2
├── models/                # Pre-trained models
├── sessions/              # Training session logs
└── assets/
    └── logo.png
```

## Implementation Details

### PPO Algorithm

All implementations use the PPO-Clip algorithm with:

| Component | Details |
|-----------|---------|
| **Network** | Actor-Critic with 64-64 hidden units, Tanh activation |
| **Advantage** | Generalized Advantage Estimation (GAE) |
| **Clipping** | Surrogate objective clipping (ε = 0.2) |
| **Gradient** | Max norm clipping at 0.5 |

### Environment-Specific Notes

**MountainCar** uses reward shaping to address sparse rewards:
- Height reward based on position
- Velocity reward to encourage momentum building
- Goal bonus for reaching the flag

## Design Philosophy

**One flat file per environment/algorithm combination.** Each script is completely self-contained:

- No imports from other project files
- Duplicate code freely between files
- Easy to read, copy, and modify independently

This intentional duplication keeps each implementation simple, portable, and easy to understand without jumping between files.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgements

- [OpenAI Gymnasium](https://gymnasium.farama.org/) — RL environments
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [PPO Paper](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017
