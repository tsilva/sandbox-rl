"""
Shared utilities for PPO training scripts.
Contains session management, checkpointing, logging, and metrics tracking.
"""

import os
import json
from datetime import datetime
import torch


# =============================================================================
# Session Management
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


def start_run(session_path: str, reason: str, diagnosis: str, config: dict,
              env_full_name: str, solve_threshold: float) -> tuple[str, str]:
    """Create run folder, log RUN_START with reasoning. Returns (run_path, run_id)."""
    run_id = get_next_run_id(session_path)
    run_path = os.path.join(session_path, run_id)
    os.makedirs(run_path, exist_ok=True)

    with open(os.path.join(run_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timesteps_per_iter = config["n_envs"] * config["n_steps"]

    log_entry = f"""
{'='*80}
[{timestamp}] RUN START: {run_id}
{'='*80}

OBJECTIVE: Solve {env_full_name} (reward >= {solve_threshold}) in minimum timesteps

REASONING: {reason}
"""
    if diagnosis:
        log_entry += f"""
DIAGNOSIS FROM PREVIOUS RUN:
{diagnosis}
"""
    log_entry += f"""
HYPERPARAMETERS:
  n_envs={config['n_envs']}, n_steps={config['n_steps']} -> {timesteps_per_iter} timesteps/iteration
  batch_size={config['batch_size']}, n_epochs={config['n_epochs']}
  lr={config['lr']}, gamma={config['gamma']}, lam={config['lam']}
  clip_eps={config['clip_eps']}, value_coef={config['value_coef']}, entropy_coef={config['entropy_coef']}
  max_iterations={config['max_iterations']} -> max {config['max_iterations'] * timesteps_per_iter:,} timesteps

"""
    log_decision(session_path, log_entry)
    return run_path, run_id


def stop_run(session_path: str, run_id: str, status: str, diagnosis: str,
             best_reward: float, iterations: int, total_timesteps: int):
    """Log RUN_STOP with results and diagnosis for next run."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"""
{'-'*80}
[{timestamp}] RUN END: {run_id}
{'-'*80}

RESULT: {status}
  Best Reward: {best_reward:.1f}
  Iterations: {iterations}
  Total Timesteps: {total_timesteps:,}

DIAGNOSIS:
{diagnosis}

"""
    log_decision(session_path, log_entry)


# =============================================================================
# Session Analysis
# =============================================================================

def get_all_runs_summary(session_path: str, solve_threshold: float) -> list[dict]:
    """Get summary of all runs in the session."""
    runs = []
    for entry in sorted(os.listdir(session_path)):
        if entry.startswith("run_"):
            run_path = os.path.join(session_path, entry)
            config_file = os.path.join(run_path, "config.json")
            metrics_file = os.path.join(run_path, "metrics.json")

            if os.path.exists(config_file):
                with open(config_file) as f:
                    config = json.load(f)

                # Calculate total timesteps from config
                timesteps_per_iter = config["n_envs"] * config["n_steps"]

                # Get final metrics
                final_reward = None
                iterations = 0
                solved = False
                if os.path.exists(metrics_file):
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                    if metrics:
                        iterations = metrics[-1]["iteration"]
                        final_reward = metrics[-1]["reward_mean"]
                        solved = final_reward >= solve_threshold

                total_timesteps = iterations * timesteps_per_iter

                runs.append({
                    "run_id": entry,
                    "config": config,
                    "iterations": iterations,
                    "total_timesteps": total_timesteps,
                    "final_reward": final_reward,
                    "solved": solved,
                })
    return runs


def find_best_run(runs: list[dict]) -> dict | None:
    """Find the best run (solved with minimum timesteps)."""
    solved_runs = [r for r in runs if r["solved"]]
    if not solved_runs:
        return None
    return min(solved_runs, key=lambda r: r["total_timesteps"])


def log_session_summary(session_path: str, solve_threshold: float):
    """Log a summary comparing all runs at the end of the session."""
    runs = get_all_runs_summary(session_path, solve_threshold)
    if not runs:
        return

    best_run = find_best_run(runs)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary = f"""
{'='*80}
[{timestamp}] SESSION SUMMARY
{'='*80}

RUNS COMPARISON:
"""
    # Sort by timesteps for comparison
    sorted_runs = sorted(runs, key=lambda r: r["total_timesteps"] if r["solved"] else float('inf'))

    for i, run in enumerate(sorted_runs, 1):
        status = "SOLVED" if run["solved"] else "NOT SOLVED"
        marker = " <-- BEST" if best_run and run["run_id"] == best_run["run_id"] else ""
        summary += f"""
  {i}. {run['run_id']}: {run['total_timesteps']:,} timesteps ({run['iterations']} iterations) - {status}{marker}
     lr={run['config']['lr']}, n_envs={run['config']['n_envs']}, n_steps={run['config']['n_steps']}
     n_epochs={run['config']['n_epochs']}, entropy_coef={run['config']['entropy_coef']}
"""

    if best_run:
        summary += f"""
{'='*80}
BEST SOLUTION: {best_run['run_id']}
{'='*80}

Solved in {best_run['total_timesteps']:,} timesteps ({best_run['iterations']} iterations)
Final reward: {best_run['final_reward']:.1f}

WINNING HYPERPARAMETERS:
  n_envs={best_run['config']['n_envs']}
  n_steps={best_run['config']['n_steps']}
  batch_size={best_run['config']['batch_size']}
  n_epochs={best_run['config']['n_epochs']}
  lr={best_run['config']['lr']}
  gamma={best_run['config']['gamma']}
  lam={best_run['config']['lam']}
  clip_eps={best_run['config']['clip_eps']}
  value_coef={best_run['config']['value_coef']}
  entropy_coef={best_run['config']['entropy_coef']}

"""
    else:
        summary += f"""
{'='*80}
NO SUCCESSFUL SOLUTION
{'='*80}
None of the runs achieved the solve threshold of {solve_threshold}.

"""

    log_decision(session_path, summary)


def save_best_hyperparams(session_path: str, config: dict, run_id: str,
                          total_timesteps: int, final_reward: float):
    """Save the best hyperparameters to a JSON file."""
    best_hp = {
        "run_id": run_id,
        "total_timesteps": total_timesteps,
        "final_reward": final_reward,
        "hyperparameters": config,
    }
    with open(os.path.join(session_path, "best_hyperparams.json"), "w") as f:
        json.dump(best_hp, f, indent=2)


# =============================================================================
# Checkpointing & Metrics
# =============================================================================

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


# =============================================================================
# Logging
# =============================================================================

def setup_run_logging(run_path: str):
    """Return a log function that writes to both console and run.log."""
    log_file = open(os.path.join(run_path, "run.log"), "w")
    def log(msg: str):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
    return log, log_file
