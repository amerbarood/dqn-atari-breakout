"""DQN training loop with experience replay and a target network."""

from __future__ import annotations

import csv
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer

from .config import Hyperparameters
from .env import make_env
from .model import QNetwork

RUNS_DIR = Path("runs")


def linear_schedule(start_e: float, end_e: float, duration: float, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def set_seed(params: Hyperparameters) -> None:
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = params.torch_deterministic


def save_checkpoint(q_network: QNetwork, params: Hyperparameters, path: Path, global_step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"params": asdict(params), "global_step": global_step, "state_dict": q_network.state_dict()}, path)


def train(params: Hyperparameters, checkpoint_every: int = 250_000) -> Path:
    """Train an agent; returns the run directory.

    Writes runs/<run_name>/returns.csv (one row per finished game),
    model checkpoints, and gameplay videos if params.capture_video.
    """
    run_name = f"{params.env_id}__{params.exp_name}__{params.seed}__{int(time.time())}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    for key, value in asdict(params).items():
        print(f"  {key}: {value}")

    set_seed(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = make_env(params.env_id, params.seed, str(run_dir / "videos") if params.capture_video else None)
    n_actions = env.action_space.n

    q_network = QNetwork(n_actions).to(device)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=params.learning_rate)
    target_network = QNetwork(n_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Experience replay memory: sampling random minibatches decorrelates the
    # transitions in a batch, which stabilises training.
    rb = ReplayBuffer(
        params.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        optimize_memory_usage=False,
        handle_timeout_termination=False,
    )

    returns_file = open(run_dir / "returns.csv", "w", newline="")
    returns_writer = csv.writer(returns_file)
    returns_writer.writerow(["global_step", "episode", "episodic_return", "episodic_length"])

    obs, _ = env.reset(seed=params.seed)
    episode_num = 0
    start = time.time()
    try:
        for global_step in range(params.total_timesteps):
            epsilon = linear_schedule(
                params.start_e, params.end_e, params.exploration_fraction * params.total_timesteps, global_step
            )
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_network(torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32, device=device))
                action = int(q_values.argmax(dim=1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)

            if "episode" in info:
                episode_num += 1
                ep_return, ep_length = float(info["episode"]["r"]), int(info["episode"]["l"])
                returns_writer.writerow([global_step, episode_num, ep_return, ep_length])
                returns_file.flush()
                print(f"global_step={global_step}, episode={episode_num}, episodic_return={ep_return}")

            # Store the transition. `next_obs` is the true next observation here
            # because we reset the environment ourselves below.
            rb.add(
                np.asarray(obs)[None], np.asarray(next_obs)[None], np.array([action]),
                np.array([reward]), np.array([terminated]), [info],
            )
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

            if global_step > params.learning_starts:
                if global_step % params.train_frequency == 0:
                    data = rb.sample(params.batch_size)
                    with torch.no_grad():
                        target_max, _ = target_network(data.next_observations.float()).max(dim=1)
                        td_target = data.rewards.flatten() + params.gamma * target_max * (1 - data.dones.flatten())
                    old_val = q_network(data.observations.float()).gather(1, data.actions).squeeze(1)
                    loss = F.mse_loss(old_val, td_target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if global_step % params.target_network_frequency == 0:
                    for target_param, q_param in zip(target_network.parameters(), q_network.parameters()):
                        target_param.data.copy_(params.tau * q_param.data + (1.0 - params.tau) * target_param.data)

            if params.save_model and global_step > 0 and global_step % checkpoint_every == 0:
                save_checkpoint(q_network, params, run_dir / f"checkpoint_{global_step}.pt", global_step)
                print(f"step {global_step}: checkpoint saved, {global_step / (time.time() - start):.0f} steps/s")
    except KeyboardInterrupt:
        print("\nInterrupted, saving final model...")
    finally:
        returns_file.close()
        if params.save_model:
            save_checkpoint(q_network, params, run_dir / "model.pt", global_step)
            print(f"Saved {run_dir / 'model.pt'}")
        env.close()
    return run_dir
