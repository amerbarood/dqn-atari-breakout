"""Fast checks: network shapes, configs, schedule, theory task and a tiny training run."""

import torch

from dqn_breakout import theory
from dqn_breakout.config import RUNS, get_run
from dqn_breakout.model import QNetwork
from dqn_breakout.train import linear_schedule


def test_qnetwork_shape():
    assert QNetwork(4)(torch.rand(2, 4, 84, 84)).shape == (2, 4)


def test_run_configs():
    assert RUNS["F"].buffer_size == 400_000 and RUNS["F"].gamma == 0.95 and RUNS["F"].batch_size == 128
    assert get_run("a", total_timesteps=10).total_timesteps == 10


def test_linear_schedule():
    assert linear_schedule(1.0, 0.01, 100, 0) == 1.0
    assert linear_schedule(1.0, 0.01, 100, 1000) == 0.01


def test_theory_values():
    values = theory.policy_evaluation()
    assert values.shape == (3, 4)
    assert values[0, 3] == 1.0 and values[0, 1] == -1.0


def test_tiny_training_run(tmp_path, monkeypatch):
    from dqn_breakout import train as train_mod

    monkeypatch.setattr(train_mod, "RUNS_DIR", tmp_path)
    params = get_run("A", total_timesteps=300, learning_starts=100, buffer_size=1000, capture_video=False)
    run_dir = train_mod.train(params)
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "returns.csv").exists()
