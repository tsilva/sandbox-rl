"""
PPO (Proximal Policy Optimization) implementation for CartPole-v1.
Solves the environment when average reward >= 475 over 100 episodes.

Usage:
    python ppo_cartpole.py          # Train and save policy
    python ppo_cartpole.py --play   # Play back saved policy with rendering
"""

import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical

from util import (
    create_session, start_run, stop_run, log_decision,
    get_all_runs_summary, find_best_run, log_session_summary,
    save_best_hyperparams, save_checkpoint, append_metrics, setup_run_logging
)

# Standardized model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ppo_cartpole.pt")

# Environment constants
ENV_NAME = "cartpole"
ENV_FULL_NAME = "CartPole-v1"
SOLVE_THRESHOLD = 475


def get_default_hyperparams() -> dict:
    """Return default hyperparameters for CartPole."""
    return {
        "n_envs": 8,
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 10,
        "lr": 3e-4,
        "gamma": 0.99,
        "lam": 0.95,
        "clip_eps": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_iterations": 500,
    }


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


def train(session_path: str = None, reason: str = "Initial training attempt",
          diagnosis: str = "", hyperparams: dict = None):
    """Main training loop with vectorized environments."""
    # Use provided hyperparams or defaults
    if hyperparams is None:
        hyperparams = get_default_hyperparams()

    n_envs = hyperparams["n_envs"]
    n_steps = hyperparams["n_steps"]
    batch_size = hyperparams["batch_size"]
    n_epochs = hyperparams["n_epochs"]
    lr = hyperparams["lr"]
    gamma = hyperparams["gamma"]
    lam = hyperparams["lam"]
    clip_eps = hyperparams["clip_eps"]
    value_coef = hyperparams["value_coef"]
    entropy_coef = hyperparams["entropy_coef"]
    max_iterations = hyperparams["max_iterations"]
    solve_threshold = SOLVE_THRESHOLD

    # Session mode setup
    run_path = None
    run_id = None
    log_file = None
    log_fn = print  # Default to print

    if session_path:
        run_path, run_id = start_run(session_path, reason, diagnosis, hyperparams,
                                      ENV_FULL_NAME, SOLVE_THRESHOLD)
        log_fn, log_file = setup_run_logging(run_path)

    # Vectorized environment
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(ENV_FULL_NAME) for _ in range(n_envs)])
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.n

    model = ActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    buffer = RolloutBuffer(n_steps, n_envs, obs_dim)

    # Tracking
    episode_rewards = []
    episode_reward_buffer = np.zeros(n_envs)
    states, _ = envs.reset()
    best_reward = float('-inf')

    log_fn(f"Starting PPO training for {ENV_FULL_NAME} (vectorized)...")
    log_fn(f"Using {n_envs} parallel environments")
    log_fn(f"Target: Average reward >= {solve_threshold} over 100 episodes")
    if run_path:
        log_fn(f"Run path: {run_path}\n")
    else:
        log_fn("")

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
            reward_std = np.std(episode_rewards[-100:])
            log_fn(f"Iteration {iteration + 1}: Avg reward (last 100 eps) = {avg_reward:.2f}")

            # Track metrics and checkpoints in session mode
            if run_path:
                append_metrics(run_path, iteration + 1, avg_reward, reward_std, len(episode_rewards))
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    save_checkpoint(run_path, model, obs_dim, act_dim, iteration + 1, avg_reward)
                    log_fn(f"  -> New best! Saved checkpoint (reward: {avg_reward:.2f})")

            if avg_reward >= solve_threshold:
                log_fn(f"\nSolved! Average reward {avg_reward:.2f} >= {solve_threshold}")
                break
        else:
            if len(episode_rewards) > 0:
                avg_reward = np.mean(episode_rewards)
                reward_std = np.std(episode_rewards)
                log_fn(f"Iteration {iteration + 1}: Avg reward ({len(episode_rewards)} eps) = {avg_reward:.2f}")

                # Track metrics in session mode
                if run_path:
                    append_metrics(run_path, iteration + 1, avg_reward, reward_std, len(episode_rewards))
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                        save_checkpoint(run_path, model, obs_dim, act_dim, iteration + 1, avg_reward)
                        log_fn(f"  -> New best! Saved checkpoint (reward: {avg_reward:.2f})")

    envs.close()
    final_iteration = iteration + 1

    # Final evaluation
    log_fn("\nRunning final evaluation...")
    eval_env = gym.make(ENV_FULL_NAME)
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
    log_fn(f"Final evaluation: {final_avg:.2f} average reward over 100 episodes")

    solved = final_avg >= solve_threshold
    if solved:
        log_fn("SUCCESS: Environment solved!")
    else:
        log_fn(f"Not yet solved (need {solve_threshold})")

    # Handle session mode cleanup
    total_timesteps = final_iteration * n_envs * n_steps
    if session_path and run_path:
        status = "SOLVED" if solved else "NOT_SOLVED"

        # Generate diagnosis for next run
        if solved:
            run_diagnosis = (
                f"Successfully solved in {total_timesteps:,} timesteps ({final_iteration} iterations). "
                f"Final avg reward: {final_avg:.1f}. Learning was stable with consistent improvement."
            )
        else:
            run_diagnosis = (
                f"Did not solve after {total_timesteps:,} timesteps. "
                f"Best reward: {best_reward:.1f}, final eval: {final_avg:.1f}. "
                f"Consider: adjusting learning rate, increasing entropy for exploration, or more training time."
            )

        stop_run(session_path, run_id, status, run_diagnosis, best_reward, final_iteration, total_timesteps)

        # Find best run across all runs and update session artifacts
        runs = get_all_runs_summary(session_path, SOLVE_THRESHOLD)
        best_run = find_best_run(runs)

        if best_run:
            best_run_path = os.path.join(session_path, best_run["run_id"])
            best_model_src = os.path.join(best_run_path, "checkpoint_best.pt")
            best_model_dst = os.path.join(session_path, "best_model.pt")

            if os.path.exists(best_model_src):
                shutil.copy(best_model_src, best_model_dst)
                log_fn(f"Best model from {best_run['run_id']} copied to: {best_model_dst}")

            # Save best hyperparameters
            save_best_hyperparams(
                session_path, best_run["config"], best_run["run_id"],
                best_run["total_timesteps"], best_run["final_reward"]
            )
            log_fn(f"Best hyperparams saved to: {os.path.join(session_path, 'best_hyperparams.json')}")

        # Log session summary comparing all runs
        log_session_summary(session_path, SOLVE_THRESHOLD)
        log_fn(f"Session summary appended to: {os.path.join(session_path, 'session.log')}")

        if log_file:
            log_file.close()
    else:
        # Legacy mode: save to standard location
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "obs_dim": int(obs_dim),
            "act_dim": int(act_dim),
        }, MODEL_PATH)
        log_fn(f"\nModel saved to: {MODEL_PATH}")

    return model


def play(num_episodes: int = 5, session_path: str = None):
    """Play back a trained policy with rendering."""
    # Determine model path
    if session_path:
        model_path = os.path.join(session_path, "best_model.pt")
        if not os.path.exists(model_path):
            print(f"No best model found at {model_path}")
            print("Session may not have completed successfully.")
            return
    else:
        model_path = MODEL_PATH
        if not os.path.exists(model_path):
            print(f"No saved model found at {model_path}")
            print("Please train first: python ppo_cartpole.py")
            return

    # Load model
    checkpoint = torch.load(model_path, weights_only=True)
    model = ActorCritic(checkpoint["obs_dim"], checkpoint["act_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from {model_path}")
    print(f"Playing {num_episodes} episodes...\n")

    # Create environment with human rendering
    env = gym.make(ENV_FULL_NAME, render_mode="human")

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
    defaults = get_default_hyperparams()

    parser = argparse.ArgumentParser(
        description=f"PPO for {ENV_FULL_NAME}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Hyperparameter overrides:
  Use --hp to override any hyperparameter: --hp lr=1e-3 --hp n_envs=16

  Available hyperparameters:
    n_envs        Number of parallel environments (default: 8)
    n_steps       Steps per env per rollout (default: 256)
    batch_size    Minibatch size for updates (default: 64)
    n_epochs      PPO epochs per iteration (default: 10)
    lr            Learning rate (default: 3e-4)
    gamma         Discount factor (default: 0.99)
    lam           GAE lambda (default: 0.95)
    clip_eps      PPO clip epsilon (default: 0.2)
    value_coef    Value loss coefficient (default: 0.5)
    entropy_coef  Entropy bonus coefficient (default: 0.01)
    max_iterations Maximum training iterations (default: 500)
"""
    )
    parser.add_argument("--play", action="store_true", help="Play back saved policy")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--session", type=str,
                        help="Path to existing session to continue (omit to create new)")
    parser.add_argument("--reason", type=str, default="Initial training attempt",
                        help="Reason for starting this run (logged in session.log)")
    parser.add_argument("--diagnosis", type=str, default="",
                        help="Diagnosis from previous run explaining what to improve")
    parser.add_argument("--hp", action="append", metavar="KEY=VALUE",
                        help="Hyperparameter override (can use multiple times)")
    args = parser.parse_args()

    # Parse hyperparameter overrides
    hyperparams = defaults.copy()
    if args.hp:
        for hp_str in args.hp:
            if "=" not in hp_str:
                print(f"Error: Invalid hyperparameter format '{hp_str}'. Use KEY=VALUE")
                exit(1)
            key, value = hp_str.split("=", 1)
            if key not in defaults:
                print(f"Error: Unknown hyperparameter '{key}'")
                print(f"Available: {', '.join(defaults.keys())}")
                exit(1)
            # Convert to appropriate type
            default_val = defaults[key]
            try:
                if isinstance(default_val, int):
                    hyperparams[key] = int(value)
                elif isinstance(default_val, float):
                    hyperparams[key] = float(value)
                else:
                    hyperparams[key] = value
            except ValueError:
                print(f"Error: Cannot convert '{value}' to {type(default_val).__name__} for '{key}'")
                exit(1)

    if args.play:
        play(args.episodes, session_path=args.session)
    else:
        # Session logging is always enabled
        if args.session:
            # Continue existing session
            session_path = args.session
            if not os.path.isdir(session_path):
                print(f"Error: Session path does not exist: {session_path}")
                exit(1)
            print(f"Continuing session: {session_path}")
        else:
            # Create new session
            session_path = create_session(ENV_NAME, ENV_FULL_NAME, SOLVE_THRESHOLD)
            print(f"Created new session: {session_path}")

        train(session_path=session_path, reason=args.reason,
              diagnosis=args.diagnosis, hyperparams=hyperparams)
