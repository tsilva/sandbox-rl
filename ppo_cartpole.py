"""
PPO (Proximal Policy Optimization) implementation for CartPole-v1.
Solves the environment when average reward >= 475 over 100 episodes.

Usage:
    python ppo_cartpole.py          # Train and save policy
    python ppo_cartpole.py --play   # Play back saved policy with rendering
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical

# Standardized model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ppo_cartpole.pt")


class ActorCritic(nn.Module):
    """Combined Actor-Critic network."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()

        # Actor: outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
            nn.Softmax(dim=-1)
        )

        # Critic: outputs state value
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor):
        return self.actor(x), self.critic(x)

    def get_action(self, states: np.ndarray):
        """Sample actions from policy for vectorized environments."""
        states_t = torch.FloatTensor(states)
        probs, values = self.forward(states_t)
        dist = Categorical(probs)
        actions = dist.sample()
        return actions.numpy(), dist.log_prob(actions).detach().numpy(), values.squeeze(-1).detach().numpy()

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for PPO update."""
        probs, values = self.forward(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


class RolloutBuffer:
    """Stores rollout data for PPO updates with vectorized environments."""

    def __init__(self, n_steps: int, n_envs: int, obs_dim: int):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.reset()

    def reset(self):
        self.states = np.zeros((self.n_steps, self.n_envs, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.n_envs), dtype=np.int64)
        self.rewards = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.ptr = 0

    def add(self, states, actions, rewards, dones, log_probs, values):
        self.states[self.ptr] = states
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr] = values
        self.ptr += 1

    def get_flattened(self):
        """Return flattened arrays for training."""
        batch_size = self.n_steps * self.n_envs
        return (
            self.states.reshape(batch_size, -1),
            self.actions.reshape(batch_size),
            self.rewards,  # Keep shape for GAE
            self.dones,    # Keep shape for GAE
            self.log_probs.reshape(batch_size),
            self.values,   # Keep shape for GAE
        )


def compute_gae_vectorized(rewards, values, dones, next_values, gamma=0.99, lam=0.95):
    """Compute GAE for vectorized environments."""
    n_steps, n_envs = rewards.shape
    advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
    last_gae = np.zeros(n_envs, dtype=np.float32)

    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_val = next_values
        else:
            next_val = values[t + 1]

        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages.flatten(), returns.flatten()


def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages,
               clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
               batch_size=64, n_epochs=10):
    """Perform PPO update."""
    n_samples = len(states)

    # Convert to tensors
    states_t = torch.FloatTensor(states)
    actions_t = torch.LongTensor(actions)
    old_log_probs_t = torch.FloatTensor(old_log_probs)
    returns_t = torch.FloatTensor(returns)
    advantages_t = torch.FloatTensor(advantages)

    # Normalize advantages
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    for _ in range(n_epochs):
        # Shuffle indices
        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            # Get batch data
            batch_states = states_t[batch_idx]
            batch_actions = actions_t[batch_idx]
            batch_old_log_probs = old_log_probs_t[batch_idx]
            batch_returns = returns_t[batch_idx]
            batch_advantages = advantages_t[batch_idx]

            # Evaluate current policy
            log_probs, values, entropy = model.evaluate(batch_states, batch_actions)

            # Compute ratio and clipped objective
            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(values, batch_returns)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            # Update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


def train():
    """Main training loop with vectorized environments."""
    # Hyperparameters
    n_envs = 8  # Number of parallel environments
    n_steps = 256  # Steps per env per rollout (total = n_envs * n_steps = 2048)
    batch_size = 64
    n_epochs = 10
    lr = 3e-4
    gamma = 0.99
    lam = 0.95
    clip_eps = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    max_iterations = 500
    solve_threshold = 475

    # Vectorized environment
    envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(n_envs)])
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.n

    model = ActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    buffer = RolloutBuffer(n_steps, n_envs, obs_dim)

    # Tracking
    episode_rewards = []
    episode_reward_buffer = np.zeros(n_envs)
    states, _ = envs.reset()

    print("Starting PPO training for CartPole-v1 (vectorized)...")
    print(f"Using {n_envs} parallel environments")
    print(f"Target: Average reward >= {solve_threshold} over 100 episodes\n")

    for iteration in range(max_iterations):
        # Collect rollouts
        buffer.reset()

        for step in range(n_steps):
            actions, log_probs, values = model.get_action(states)
            next_states, rewards, terminations, truncations, infos = envs.step(actions)
            dones = np.logical_or(terminations, truncations)

            buffer.add(states, actions, rewards, dones.astype(np.float32), log_probs, values)
            episode_reward_buffer += rewards

            # Handle episode completions
            for i, done in enumerate(dones):
                if done:
                    episode_rewards.append(episode_reward_buffer[i])
                    episode_reward_buffer[i] = 0

            states = next_states

        # Get final values for GAE
        with torch.no_grad():
            _, next_values = model.forward(torch.FloatTensor(states))
            next_values = next_values.squeeze(-1).numpy()

        # Get buffer data
        flat_states, flat_actions, rewards, dones_arr, flat_log_probs, values = buffer.get_flattened()

        # Compute GAE
        advantages, returns = compute_gae_vectorized(rewards, values, dones_arr, next_values, gamma, lam)

        # PPO update
        ppo_update(model, optimizer, flat_states, flat_actions, flat_log_probs, returns, advantages,
                   clip_eps, value_coef, entropy_coef, batch_size, n_epochs)

        # Evaluate
        if len(episode_rewards) >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Iteration {iteration + 1}: Avg reward (last 100 eps) = {avg_reward:.2f}")

            if avg_reward >= solve_threshold:
                print(f"\nSolved! Average reward {avg_reward:.2f} >= {solve_threshold}")
                break
        else:
            if len(episode_rewards) > 0:
                avg_reward = np.mean(episode_rewards)
                print(f"Iteration {iteration + 1}: Avg reward ({len(episode_rewards)} eps) = {avg_reward:.2f}")

    envs.close()

    # Final evaluation
    print("\nRunning final evaluation...")
    eval_env = gym.make("CartPole-v1")
    eval_rewards = []

    for _ in range(100):
        state, _ = eval_env.reset()
        ep_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                probs, _ = model.forward(torch.FloatTensor(state).unsqueeze(0))
                action = torch.argmax(probs).item()
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_reward += reward

        eval_rewards.append(ep_reward)

    eval_env.close()

    final_avg = np.mean(eval_rewards)
    print(f"Final evaluation: {final_avg:.2f} average reward over 100 episodes")

    if final_avg >= solve_threshold:
        print("SUCCESS: Environment solved!")
    else:
        print(f"Not yet solved (need {solve_threshold})")

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
    }, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    return model


def play(num_episodes: int = 5):
    """Play back a trained policy with rendering."""
    if not os.path.exists(MODEL_PATH):
        print(f"No saved model found at {MODEL_PATH}")
        print("Please train first: python ppo_cartpole.py")
        return

    # Load model
    checkpoint = torch.load(MODEL_PATH, weights_only=True)
    model = ActorCritic(checkpoint["obs_dim"], checkpoint["act_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from {MODEL_PATH}")
    print(f"Playing {num_episodes} episodes...\n")

    # Create environment with human rendering
    env = gym.make("CartPole-v1", render_mode="human")

    for ep in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                probs, _ = model.forward(torch.FloatTensor(state).unsqueeze(0))
                action = torch.argmax(probs).item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        print(f"Episode {ep + 1}: Reward = {ep_reward:.0f}")

    env.close()
    print("\nPlayback complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO for CartPole-v1")
    parser.add_argument("--play", action="store_true", help="Play back saved policy")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    args = parser.parse_args()

    if args.play:
        play(args.episodes)
    else:
        train()
