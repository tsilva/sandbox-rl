# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning sandbox implementing PPO (Proximal Policy Optimization) for Gymnasium environments. Each environment has a standalone, self-contained Python script with no shared code between implementations.

## Commands

```bash
# Install dependencies
pip install torch numpy gymnasium

# Train models
python ppo_cartpole.py          # Train CartPole agent
python ppo_mountaincar.py       # Train MountainCar agent

# Play with trained models
python ppo_cartpole.py --play              # Watch trained agent
python ppo_cartpole.py --play --episodes 10  # Specify episode count
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
