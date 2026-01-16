<p align="center">
  <img src="assets/logo.png" alt="sandbox-rl logo" width="200">
</p>

# sandbox-rl

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0081A5?logo=openaigym&logoColor=white)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-DA7857?logo=anthropic)](https://claude.ai/code)

## Overview

A sandbox for reinforcement learning experiments using Proximal Policy Optimization (PPO). This repository contains clean, self-contained implementations of PPO for classic control environments from OpenAI Gymnasium.

Each implementation features:
- Actor-Critic neural network architecture
- Generalized Advantage Estimation (GAE)
- Vectorized environments for parallel data collection
- Train and play modes with pre-trained models included

## Environments

| Environment | Solve Threshold | Description |
|-------------|-----------------|-------------|
| CartPole-v1 | >= 475 avg reward | Balance a pole on a moving cart |
| MountainCar-v0 | >= -110 avg reward | Drive a car up a steep hill |

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/tsilva/sandbox-rl.git
cd sandbox-rl
pip install torch numpy gymnasium
```

### Usage

**Train a new model:**

```bash
python ppo_cartpole.py          # Train CartPole agent
python ppo_mountaincar.py       # Train MountainCar agent
```

**Play with pre-trained model:**

```bash
python ppo_cartpole.py --play              # Watch CartPole agent
python ppo_mountaincar.py --play           # Watch MountainCar agent
python ppo_cartpole.py --play --episodes 10  # Specify number of episodes
```

## Repository Structure

```
sandbox-rl/
├── ppo_cartpole.py       # PPO implementation for CartPole-v1
├── ppo_mountaincar.py    # PPO implementation for MountainCar-v0
├── models/
│   ├── ppo_cartpole.pt   # Pre-trained CartPole model
│   └── ppo_mountaincar.pt # Pre-trained MountainCar model
└── README.md
```

## Implementation Details

### PPO Algorithm

Both implementations use the PPO-Clip algorithm with:

- **Network**: Separate actor and critic heads with shared input processing (64-64 hidden units, Tanh activation)
- **GAE**: Lambda = 0.95, Gamma = 0.99
- **PPO Updates**: 10 epochs, batch size 64, clip epsilon 0.2
- **Optimization**: Adam with lr=3e-4, gradient clipping at 0.5

### MountainCar Reward Shaping

MountainCar uses reward shaping to address the sparse reward problem:
- Height reward based on position
- Velocity reward to encourage momentum
- Goal bonus for reaching the flag

## Reporting Issues

Found a bug or have a suggestion? Please open an issue:

[GitHub Issues](https://github.com/tsilva/sandbox-rl/issues)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenAI Gymnasium](https://gymnasium.farama.org/) - Reinforcement learning environments
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017

## Contact

**Tiago Silva**

- GitHub: [@tsilva](https://github.com/tsilva)
- LinkedIn: [tsilva.eu](https://tsilva.eu)
