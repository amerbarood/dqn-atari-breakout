"""Hyperparameters for the six experiment runs (A-F).

Values are the ones used for the six training runs in the study.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class Hyperparameters:
    env_id: str = "BreakoutNoFrameskip-v4"
    exp_name: str = "DQN_Breakout"
    seed: int = 1
    torch_deterministic: bool = True
    capture_video: bool = True
    save_model: bool = True

    total_timesteps: int = 10_000_000
    learning_rate: float = 1e-4
    buffer_size: int = 100_000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 1000
    batch_size: int = 32
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.1
    learning_starts: int = 80_000
    train_frequency: int = 4


_BASE = Hyperparameters()

RUNS: dict[str, Hyperparameters] = {
    # Baseline, buffer limited to 100k by memory.
    "A": _BASE,
    "B": replace(_BASE, total_timesteps=5_000_000, buffer_size=200_000, end_e=0.05, exploration_fraction=0.2),
    "C": replace(_BASE, buffer_size=250_000),
    "D": replace(_BASE, buffer_size=250_000, exploration_fraction=0.333),
    "E": replace(_BASE, buffer_size=200_000, start_e=0.9, end_e=0.05, exploration_fraction=0.08),
    # Most stable run.
    "F": replace(_BASE, buffer_size=400_000, gamma=0.95, batch_size=128, exploration_fraction=0.05),
}


def get_run(name: str, **overrides) -> Hyperparameters:
    params = RUNS[name.upper()]
    params = replace(params, exp_name=f"{params.exp_name}__{name.upper()}")
    overrides = {k: v for k, v in overrides.items() if v is not None}
    return replace(params, **overrides)
