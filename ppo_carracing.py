#!/usr/bin/env python3
"""
PPO implementation for CarRacing-v2 environment.
Self-contained single-file implementation.

Usage:
    python ppo_carracing.py          # Train
    python ppo_carracing.py --play   # Watch trained agent
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging to both console and file."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"carracing_{timestamp}.log")

    # Create logger
    logger = logging.getLogger("carracing")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers

    # Console handler with immediate flush
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger, log_file


def log(logger: logging.Logger, message: str):
    """Log message and flush immediately."""
    logger.info(message)
    for handler in logger.handlers:
        handler.flush()


# ============================================================================
# Preprocessing Utilities
# ============================================================================

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess a single frame:
    - Convert RGB to grayscale
    - Crop to remove dashboard (bottom 12 pixels)
    - Resize to 84x84
    - Normalize to [0, 1]
    """
    # Convert to grayscale using luminosity method
    gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])

    # Crop bottom dashboard (keep top 84 rows of original 96)
    cropped = gray[:84, :]

    # Resize to 84x84 using simple nearest-neighbor (fast)
    # The width is 96, we need to resize to 84
    # Use step-based sampling for speed
    h_indices = np.linspace(0, cropped.shape[0] - 1, 84).astype(int)
    w_indices = np.linspace(0, cropped.shape[1] - 1, 84).astype(int)
    resized = cropped[h_indices][:, w_indices]

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    return normalized


def preprocess_frame_vectorized(frames: np.ndarray) -> np.ndarray:
    """
    Preprocess a batch of frames from vectorized environments.
    Input: (n_envs, 96, 96, 3) uint8
    Output: (n_envs, 84, 84) float32
    """
    # Convert to grayscale
    gray = np.dot(frames[..., :3], [0.299, 0.587, 0.114])

    # Crop bottom dashboard
    cropped = gray[:, :84, :]

    # Resize to 84x84
    h_indices = np.linspace(0, cropped.shape[1] - 1, 84).astype(int)
    w_indices = np.linspace(0, cropped.shape[2] - 1, 84).astype(int)
    resized = cropped[:, h_indices][:, :, w_indices]

    # Normalize
    normalized = resized.astype(np.float32) / 255.0

    return normalized


# ============================================================================
# Frame Stack Wrapper
# ============================================================================

class FrameStack:
    """
    Stacks the last n_stack preprocessed frames.
    Works with vectorized environments.
    """

    def __init__(self, n_envs: int, n_stack: int = 4, frame_shape: tuple = (84, 84)):
        self.n_envs = n_envs
        self.n_stack = n_stack
        self.frame_shape = frame_shape
        self.frames = np.zeros((n_envs, n_stack, *frame_shape), dtype=np.float32)

    def reset(self, first_frames: np.ndarray, env_indices: np.ndarray = None):
        """
        Reset frame stack with initial frames.
        first_frames: (n_envs, 84, 84) or (subset, 84, 84)
        env_indices: which environments to reset (None = all)
        """
        if env_indices is None:
            # Reset all environments
            for i in range(self.n_stack):
                self.frames[:, i] = first_frames
        else:
            # Reset specific environments
            for i in range(self.n_stack):
                self.frames[env_indices, i] = first_frames

    def add_frame(self, new_frames: np.ndarray):
        """
        Add new frames and shift the stack.
        new_frames: (n_envs, 84, 84)
        """
        self.frames = np.roll(self.frames, shift=-1, axis=1)
        self.frames[:, -1] = new_frames

    def get_stacked(self) -> np.ndarray:
        """Return stacked frames: (n_envs, n_stack, 84, 84)"""
        return self.frames.copy()


# ============================================================================
# Environment Wrapper for Action Repeat
# ============================================================================

class CarRacingWrapper(gym.Wrapper):
    """
    Wrapper for CarRacing that handles:
    - Skipping initial zoom frames
    - Action repeat (frame skip)
    - Reward clipping
    """

    def __init__(self, env, skip_initial: int = 50, action_repeat: int = 4):
        super().__init__(env)
        self.skip_initial = skip_initial
        self.action_repeat = action_repeat

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Skip initial zoom animation frames
        for _ in range(self.skip_initial):
            obs, _, terminated, truncated, info = self.env.step([0, 0, 0])
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)

        return obs, info

    def step(self, action):
        total_reward = 0
        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


# ============================================================================
# Actor-Critic Network with CNN
# ============================================================================

class CNNActorCritic(nn.Module):
    """
    CNN-based Actor-Critic network for image observations.
    Uses Gaussian distribution for continuous actions.
    Output: [steering, gas, brake] where:
      - steering: tanh -> [-1, 1]
      - gas: sigmoid -> [0, 1]
      - brake: sigmoid -> [0, 1]
    """

    def __init__(self, n_stack: int = 4, action_dim: int = 3):
        super().__init__()

        self.action_dim = action_dim

        # Shared CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(n_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

        # Actor head: outputs mean for each action
        self.actor_mean = nn.Linear(512, action_dim)

        # Learnable log standard deviation (clamped to prevent divergence)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = nn.Linear(512, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Smaller initialization for actor mean
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        # Bias initialization: [0, 0.5, 0] means steering=0, gas=0.5, brake=0
        # This encourages initial forward motion
        self.actor_mean.bias.data = torch.tensor([0.0, 0.5, -1.0])

        # Standard initialization for critic output
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        x: (batch, n_stack, 84, 84)
        Returns: action_mean, action_std, value
        """
        features = self.encoder(x)

        # Actor: get mean
        action_mean = self.actor_mean(features)

        # Get std from learnable parameter (clamped for stability)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(torch.clamp(action_logstd, -2.0, 0.0))  # Tighter clamp to prevent too much randomness

        # Critic
        value = self.critic(features)

        return action_mean, action_std, value

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        """
        Get action, log probability, entropy, and value.
        Actions are in raw space (before squashing).
        """
        action_mean, action_std, value = self.forward(x)

        # Create Normal distribution
        dist = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            # Sample action
            action_raw = dist.sample()
        else:
            action_raw = action

        # Log probability (sum over action dimensions)
        log_prob = dist.log_prob(action_raw).sum(dim=-1)

        # Entropy (sum over action dimensions)
        entropy = dist.entropy().sum(dim=-1)

        return action_raw, log_prob, entropy, value.squeeze(-1)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value estimate only."""
        features = self.encoder(x)
        return self.critic(features).squeeze(-1)


# ============================================================================
# Action Scaling
# ============================================================================

def scale_action(action_raw: np.ndarray) -> np.ndarray:
    """
    Scale raw Gaussian actions to environment action space.
    action_raw: (n_envs, 3) or (3,) - raw outputs from Gaussian
    Returns: scaled actions for environment
        - steering: tanh -> [-1, 1]
        - gas: sigmoid -> [0, 1]
        - brake: sigmoid -> [0, 1]
    """
    scaled = np.zeros_like(action_raw)
    # Steering: tanh to get [-1, 1]
    scaled[..., 0] = np.tanh(action_raw[..., 0])
    # Gas: sigmoid to get [0, 1]
    scaled[..., 1] = 1.0 / (1.0 + np.exp(-action_raw[..., 1]))
    # Brake: sigmoid to get [0, 1], but reduce braking tendency
    scaled[..., 2] = 1.0 / (1.0 + np.exp(-action_raw[..., 2]))
    # Reduce brake when gas is applied to avoid simultaneous gas+brake
    scaled[..., 2] = scaled[..., 2] * (1.0 - scaled[..., 1])
    return scaled


# ============================================================================
# Rollout Buffer
# ============================================================================

class RolloutBuffer:
    """Stores rollout data for PPO training."""

    def __init__(self, n_steps: int, n_envs: int, n_stack: int, frame_shape: tuple, action_dim: int, device: torch.device):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device

        # Storage
        self.observations = np.zeros((n_steps, n_envs, n_stack, *frame_shape), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)

        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)

        self.ptr = 0

    def add(self, obs, action, log_prob, reward, done, value):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def reset(self):
        self.ptr = 0

    def compute_gae(self, last_value: np.ndarray, gamma: float, gae_lambda: float):
        """Compute GAE advantages and returns."""
        last_gae = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """Generate random minibatches for training."""
        total_size = self.n_steps * self.n_envs
        indices = np.random.permutation(total_size)

        # Flatten data
        obs_flat = self.observations.reshape(total_size, *self.observations.shape[2:])
        actions_flat = self.actions.reshape(total_size, -1)
        log_probs_flat = self.log_probs.reshape(total_size)
        advantages_flat = self.advantages.reshape(total_size)
        returns_flat = self.returns.reshape(total_size)

        for start in range(0, total_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield (
                torch.tensor(obs_flat[batch_indices], device=self.device),
                torch.tensor(actions_flat[batch_indices], device=self.device),
                torch.tensor(log_probs_flat[batch_indices], device=self.device),
                torch.tensor(advantages_flat[batch_indices], device=self.device),
                torch.tensor(returns_flat[batch_indices], device=self.device),
            )


# ============================================================================
# PPO Update
# ============================================================================

def ppo_update(
    model: CNNActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    n_epochs: int,
    batch_size: int,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
):
    """Perform PPO update."""
    total_pg_loss = 0
    total_vf_loss = 0
    total_entropy = 0
    n_updates = 0

    for epoch in range(n_epochs):
        for obs, actions, old_log_probs, advantages, returns in buffer.get_batches(batch_size):
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Get current policy values
            _, new_log_probs, entropy, new_values = model.get_action_and_value(obs, actions)

            # Policy loss with clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            vf_loss = ((new_values - returns) ** 2).mean()

            # Entropy bonus
            entropy_loss = entropy.mean()

            # Total loss
            loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy_loss

            # Update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_pg_loss += pg_loss.item()
            total_vf_loss += vf_loss.item()
            total_entropy += entropy_loss.item()
            n_updates += 1

    return total_pg_loss / n_updates, total_vf_loss / n_updates, total_entropy / n_updates


# ============================================================================
# Training
# ============================================================================

def train():
    """Main training loop."""
    # Setup logging
    logger, log_file = setup_logging()

    # Hyperparameters
    n_envs = 8
    n_steps = 256  # Increased for more data per update
    n_epochs = 10
    batch_size = 512
    learning_rate = 2.5e-4  # Moderate learning rate
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.1  # Tighter clipping for stability
    vf_coef = 0.5
    ent_coef = 0.01  # Entropy coefficient
    max_grad_norm = 0.5
    max_iterations = 2000
    anneal_lr = True  # Anneal learning rate
    n_stack = 4
    frame_shape = (84, 84)
    action_dim = 3

    # CarRacing specific
    action_repeat = 4  # Repeat each action for N frames
    skip_initial_frames = 50  # Skip zoom animation

    # Solved threshold
    solved_threshold = 900

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(logger, f"Using device: {device}")
    log(logger, f"Logging to: {log_file}")

    # Create vectorized environments with wrappers
    def make_env(skip_init, act_repeat):
        def _init():
            env = gym.make("CarRacing-v2", continuous=True)
            env = CarRacingWrapper(env, skip_initial=skip_init, action_repeat=act_repeat)
            return env
        return _init

    envs = gym.vector.SyncVectorEnv([
        make_env(skip_initial_frames, action_repeat) for _ in range(n_envs)
    ])

    # Initialize frame stacker
    frame_stack = FrameStack(n_envs, n_stack, frame_shape)

    # Initialize model and optimizer
    model = CNNActorCritic(n_stack, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)

    # Initialize buffer
    buffer = RolloutBuffer(n_steps, n_envs, n_stack, frame_shape, action_dim, device)

    # Tracking
    episode_rewards = []
    episode_lengths = []
    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs)

    # Reset environments
    obs_raw, _ = envs.reset()
    obs_processed = preprocess_frame_vectorized(obs_raw)
    frame_stack.reset(obs_processed)
    obs = frame_stack.get_stacked()

    log(logger, "=" * 60)
    log(logger, "Starting CarRacing-v2 PPO Training")
    log(logger, "=" * 60)
    log(logger, f"  n_envs: {n_envs}")
    log(logger, f"  n_steps: {n_steps}")
    log(logger, f"  batch_size: {batch_size}")
    log(logger, f"  learning_rate: {learning_rate}")
    log(logger, f"  action_repeat: {action_repeat}")
    log(logger, f"  skip_initial_frames: {skip_initial_frames}")
    log(logger, f"  Target reward: {solved_threshold}")
    log(logger, "=" * 60)

    start_time = time.time()

    global_step = 0

    for iteration in range(1, max_iterations + 1):
        # Collect rollout
        buffer.reset()

        for step in range(n_steps):
            global_step += n_envs

            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, device=device)
                action_01, log_prob, _, value = model.get_action_and_value(obs_tensor)
                action_01 = action_01.cpu().numpy()
                log_prob = log_prob.cpu().numpy()
                value = value.cpu().numpy()

            # Scale action for environment
            action_scaled = scale_action(action_01)

            # Step environment
            obs_raw_next, reward, terminated, truncated, info = envs.step(action_scaled)
            done = terminated | truncated

            # Store in buffer (store action in [0,1] space)
            buffer.add(obs, action_01, log_prob, reward, done, value)

            # Track episode stats
            current_rewards += reward
            current_lengths += 1

            # Handle episode ends
            for i, d in enumerate(done):
                if d:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    current_rewards[i] = 0
                    current_lengths[i] = 0

            # Process next observation
            obs_processed = preprocess_frame_vectorized(obs_raw_next)

            # Reset frame stack for done environments
            done_indices = np.where(done)[0]
            if len(done_indices) > 0:
                frame_stack.reset(obs_processed[done_indices], done_indices)

            # Add new frame for all envs
            frame_stack.add_frame(obs_processed)
            obs = frame_stack.get_stacked()

        # Compute advantages
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=device)
            last_value = model.get_value(obs_tensor).cpu().numpy()
        buffer.compute_gae(last_value, gamma, gae_lambda)

        # Learning rate annealing
        if anneal_lr:
            frac = 1.0 - (iteration - 1) / max_iterations
            lr_now = frac * learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_now

        # PPO update
        pg_loss, vf_loss, entropy = ppo_update(
            model, optimizer, buffer, n_epochs, batch_size,
            clip_coef, vf_coef, ent_coef, max_grad_norm
        )

        # Logging
        if iteration % 10 == 0 or len(episode_rewards) >= 10:
            if len(episode_rewards) > 0:
                recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
                avg_reward = np.mean(recent_rewards)
                elapsed = time.time() - start_time
                fps = global_step / elapsed if elapsed > 0 else 0

                log(logger, f"Iter {iteration:4d} | Steps {global_step:8d} | "
                      f"Reward: {avg_reward:7.1f} | "
                      f"Eps: {len(episode_rewards):4d} | "
                      f"PG: {pg_loss:.4f} | VF: {vf_loss:.4f} | "
                      f"Ent: {entropy:.4f} | FPS: {fps:.0f}")

                # Check if solved
                if avg_reward >= solved_threshold and len(recent_rewards) >= 100:
                    log(logger, "")
                    log(logger, f"Environment SOLVED with average reward {avg_reward:.1f}!")
                    break

    # Save model
    os.makedirs("models", exist_ok=True)
    save_path = "models/ppo_carracing.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_stack": n_stack,
        "action_dim": action_dim,
    }, save_path)

    total_time = time.time() - start_time
    log(logger, "=" * 60)
    log(logger, f"Training complete!")
    log(logger, f"  Total time: {total_time / 60:.1f} minutes")
    log(logger, f"  Total steps: {global_step}")
    log(logger, f"  Total episodes: {len(episode_rewards)}")
    if len(episode_rewards) > 0:
        log(logger, f"  Final avg reward (last 100): {np.mean(episode_rewards[-100:]):.1f}")
    log(logger, f"  Model saved to: {save_path}")
    log(logger, f"  Log saved to: {log_file}")
    log(logger, "=" * 60)

    envs.close()

    return model


# ============================================================================
# Play / Evaluation
# ============================================================================

def play(episodes: int = 5):
    """Load trained model and play episodes with rendering."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    save_path = "models/ppo_carracing.pt"
    if not os.path.exists(save_path):
        print(f"No saved model found at {save_path}. Train first.")
        return

    checkpoint = torch.load(save_path, map_location=device)
    n_stack = checkpoint["n_stack"]
    action_dim = checkpoint["action_dim"]

    model = CNNActorCritic(n_stack, action_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from {save_path}")

    # Create environment with rendering
    env = gym.make("CarRacing-v2", continuous=True, render_mode="human")

    # Frame stack for single environment
    frame_stack = FrameStack(1, n_stack, (84, 84))

    for episode in range(1, episodes + 1):
        obs_raw, _ = env.reset()
        obs_processed = preprocess_frame(obs_raw)[np.newaxis]  # Add batch dim
        frame_stack.reset(obs_processed)

        total_reward = 0
        done = False
        step_count = 0

        while not done:
            obs = frame_stack.get_stacked()

            with torch.no_grad():
                obs_tensor = torch.tensor(obs, device=device)
                action_01, _, _, _ = model.get_action_and_value(obs_tensor)
                action_01 = action_01.cpu().numpy()[0]  # Remove batch dim

            action_scaled = scale_action(action_01)

            obs_raw, reward, terminated, truncated, info = env.step(action_scaled)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

            obs_processed = preprocess_frame(obs_raw)[np.newaxis]
            frame_stack.add_frame(obs_processed)

        print(f"Episode {episode}: Reward = {total_reward:.1f}, Steps = {step_count}")

    env.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO for CarRacing-v2")
    parser.add_argument("--play", action="store_true", help="Play with trained model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    args = parser.parse_args()

    if args.play:
        play(args.episodes)
    else:
        train()
