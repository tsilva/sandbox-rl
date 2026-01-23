<div align="center">
  <img src="assets/logo.png" alt="sandbox-rl" width="512"/>

  # sandbox-rl

  [![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
  [![Gymnasium](https://img.shields.io/badge/Gymnasium-0081A5?logo=openaigym&logoColor=white)](https://gymnasium.farama.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

  **🎮 Clean, self-contained PPO implementations for learning reinforcement learning**

  [Gymnasium Docs](https://gymnasium.farama.org/) · [PPO Paper](https://arxiv.org/abs/1707.06347)
</div>

## Overview

A sandbox for reinforcement learning experiments using Proximal Policy Optimization (PPO). Each implementation is a single, self-contained file that you can read, run, and modify independently.

**Features:**
- Actor-Critic neural network architecture
- Generalized Advantage Estimation (GAE)
- Vectorized environments for parallel data collection
- Train and play modes with pre-trained models included

## Environments

| Environment | Script | Solve Threshold | Status |
|-------------|--------|-----------------|--------|
| CartPole-v1 | `ppo_cartpole.py` | >= 475 avg reward | Solved |
| MountainCar-v0 | `ppo_mountaincar.py` | >= -110 avg reward | Solved |
| LunarLander-v3 | `ppo_lunarlander.py` | >= 200 avg reward | Solved |
| CarRacing-v2 | `ppo_carracing.py` | >= 900 avg reward | In Progress |

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

**Train a new model:**

```bash
python ppo_cartpole.py          # Train CartPole agent
python ppo_mountaincar.py       # Train MountainCar agent
python ppo_lunarlander.py       # Train LunarLander agent
python ppo_carracing.py         # Train CarRacing agent
```

**Play with trained models:**

```bash
python ppo_cartpole.py --play              # Watch CartPole agent
python ppo_mountaincar.py --play           # Watch MountainCar agent
python ppo_lunarlander.py --play           # Watch LunarLander agent
python ppo_cartpole.py --play --episodes 10  # Specify number of episodes
```

## Repository Structure

```
sandbox-rl/
├── ppo_cartpole.py        # PPO for CartPole-v1
├── ppo_mountaincar.py     # PPO for MountainCar-v0
├── ppo_lunarlander.py     # PPO for LunarLander-v3
├── ppo_carracing.py       # PPO for CarRacing-v2
├── models/
│   ├── ppo_cartpole.pt    # Pre-trained CartPole model
│   ├── ppo_mountaincar.pt # Pre-trained MountainCar model
│   └── ppo_lunarlander.pt # Pre-trained LunarLander model
└── assets/
    └── logo.png
```

## Implementation Details

### PPO Algorithm

All implementations use the PPO-Clip algorithm with:

| Parameter | Value |
|-----------|-------|
| Network | 64-64 hidden units, Tanh activation |
| GAE Lambda | 0.95 |
| Gamma | 0.99 |
| PPO Epochs | 10 |
| Batch Size | 64 |
| Clip Epsilon | 0.2 |
| Learning Rate | 3e-4 |
| Gradient Clipping | 0.5 |

### Reward Shaping

MountainCar uses reward shaping to address the sparse reward problem:
- Height reward based on position
- Velocity reward to encourage momentum
- Goal bonus for reaching the flag

## Design Philosophy

**One flat file per environment/algorithm combination.** Each script is completely self-contained with no imports from other project files. This makes each implementation easy to read, copy, and modify independently.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgements

- [OpenAI Gymnasium](https://gymnasium.farama.org/) - RL environments
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
