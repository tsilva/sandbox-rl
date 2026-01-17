"""
PPO (Proximal Policy Optimization) implementation for LunarLander-v3.
Solves the environment when average reward >= 200 over 100 episodes.

Usage:
    python ppo_lunarlander.py          # Train and save policy
    python ppo_lunarlander.py --play   # Play back saved policy with rendering
"""

import argparse
import os
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.distributions import Categorical

# Standardized model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ppo_lunarlander.pt")

# Environment constants
ENV_NAME = "lunarlander"
ENV_FULL_NAME = "LunarLander-v3"
SOLVE_THRESHOLD = 200


# =============================================================================
# Session/Run Management Functions
# =============================================================================

def create_session(env_name: str, env_full_name: str, solve_threshold: float,
                   base_dir: str = "sessions") -> str:
    """Create new session folder and initialize decision log."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"{env_name}_{timestamp}"
    session_path = os.path.join(os.path.dirname(__file__), base_dir, session_name)
    os.makedirs(session_path, exist_ok=True)

    with open(os.path.join(session_path, "session.log"), "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"SESSION: {env_name} | Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Environment: {env_full_name} | Solve Threshold: {solve_threshold}\n")
        f.write("=" * 80 + "\n\n")
    return session_path


def get_next_run_id(session_path: str) -> str:
    """Get next sequential run ID."""
    existing = [d for d in os.listdir(session_path) if d.startswith("run_")]
    return f"run_{len(existing) + 1:03d}"


def log_decision(session_path: str, message: str):
    """Append entry to session.log."""
    with open(os.path.join(session_path, "session.log"), "a") as f:
        f.write(message + "\n")


def start_run(session_path: str, reason: str, config: dict) -> tuple[str, str]:
    """Create run folder, log RUN_START, save config.json. Returns (run_path, run_id)."""
    run_id = get_next_run_id(session_path)
    run_path = os.path.join(session_path, run_id)
    os.makedirs(run_path, exist_ok=True)

    with open(os.path.join(run_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config_str = ", ".join(f"{k}={v}" for k, v in config.items())
    log_decision(session_path, f"[{timestamp}] RUN_START {run_id}\n  Reason: {reason}\n  Config: {config_str}\n")

    return run_path, run_id


def stop_run(session_path: str, run_id: str, status: str, reason: str,
             best_reward: float, iterations: int):
    """Log RUN_STOP with final metrics."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_decision(session_path,
        f"[{timestamp}] RUN_STOP {run_id}\n"
        f"  Status: {status}\n"
        f"  Reason: {reason}\n"
        f"  Best Reward: {best_reward:.1f} | Iterations: {iterations}\n")


def save_checkpoint(run_path: str, model, obs_dim: int, act_dim: int,
                    iteration: int, reward: float):
    """Save checkpoint on improvement."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "iteration": iteration,
        "reward": reward,
    }, os.path.join(run_path, "checkpoint_best.pt"))


def append_metrics(run_path: str, iteration: int, reward_mean: float,
                   reward_std: float, num_episodes: int):
    """Append metrics to metrics.json."""
    metrics_file = os.path.join(run_path, "metrics.json")
    metrics = []
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)
    metrics.append({
        "iteration": iteration,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "num_episodes": num_episodes,
        "timestamp": datetime.now().isoformat()
    })
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)


def setup_run_logging(run_path: str):
    """Return a log function that writes to both console and run.log."""
    log_file = open(os.path.join(run_path, "run.log"), "w")
    def log(msg: str):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
    return log, log_file


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


def train(session_path: str = None, reason: str = "Initial training attempt"):
    """Main training loop with vectorized environments."""
    # Hyperparameters
    n_envs = 16  # More parallel environments for complex control
    n_steps = 256  # Longer rollouts (total batch = 4096)
    batch_size = 64
    n_epochs = 10
    lr = 3e-4
    gamma = 0.99
    lam = 0.95
    clip_eps = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    max_iterations = 1000
    solve_threshold = SOLVE_THRESHOLD

    # Session mode setup
    run_path = None
    run_id = None
    log_file = None
    log_fn = print  # Default to print

    if session_path:
        config = {
            "n_envs": n_envs, "n_steps": n_steps, "batch_size": batch_size,
            "n_epochs": n_epochs, "lr": lr, "gamma": gamma, "lam": lam,
            "clip_eps": clip_eps, "value_coef": value_coef,
            "entropy_coef": entropy_coef, "max_iterations": max_iterations
        }
        run_path, run_id = start_run(session_path, reason, config)
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
    if session_path and run_path:
        status = "SOLVED" if solved else "COMPLETED"
        reason = f"Reached solve threshold {solve_threshold}" if solved else f"Completed {final_iteration} iterations"
        stop_run(session_path, run_id, status, reason, best_reward, final_iteration)

        if solved:
            # Copy best model to session root
            import shutil
            best_model_src = os.path.join(run_path, "checkpoint_best.pt")
            best_model_dst = os.path.join(session_path, "best_model.pt")
            if os.path.exists(best_model_src):
                shutil.copy(best_model_src, best_model_dst)
                log_fn(f"Best model copied to: {best_model_dst}")

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
            print("Please train first: python ppo_lunarlander.py")
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
    parser = argparse.ArgumentParser(description=f"PPO for {ENV_FULL_NAME}")
    parser.add_argument("--play", action="store_true", help="Play back saved policy")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--session", type=str, nargs="?", const="NEW",
                        help="Session mode: omit path for new session, provide path to continue")
    parser.add_argument("--reason", type=str, default="Initial training attempt",
                        help="Reason for starting this run (logged in session.log)")
    args = parser.parse_args()

    if args.play:
        play(args.episodes, session_path=args.session if args.session and args.session != "NEW" else None)
    else:
        if args.session:
            if args.session == "NEW":
                session_path = create_session(ENV_NAME, ENV_FULL_NAME, SOLVE_THRESHOLD)
                print(f"Created new session: {session_path}")
            else:
                session_path = args.session
                if not os.path.isdir(session_path):
                    print(f"Error: Session path does not exist: {session_path}")
                    exit(1)
                print(f"Continuing session: {session_path}")
            train(session_path=session_path, reason=args.reason)
        else:
            train()
