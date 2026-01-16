# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **learning/testing sandbox** for reinforcement learning on Gymnasium environments.

### Typical Workflow

When a user asks to **"solve"** an environment:

1. **Write a single self-contained file** for the env/algorithm combo (e.g., `ppo_cartpole.py`)
2. **Check the Gymnasium docs** to find the reward threshold that defines "solved" for that environment
3. **Run the training script** and monitor progress
4. **If errors occur or training stalls**, debug and iterate — keep modifying and re-running until the environment is solved
5. **Once the reward threshold is reached**, run the script with `--play` to playback the trained policy for the user

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

# Train models
python ppo_cartpole.py          # Train CartPole agent
python ppo_mountaincar.py       # Train MountainCar agent
python ppo_lunarlander.py       # Train LunarLander agent

# Play with trained models
python ppo_cartpole.py --play              # Watch trained agent
python ppo_cartpole.py --play --episodes 10  # Specify episode count
python ppo_lunarlander.py --play           # Watch LunarLander agent
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
