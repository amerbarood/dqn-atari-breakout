"""Play games with a trained agent and plot training curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from .env import make_env  # noqa: E402
from .model import QNetwork  # noqa: E402


def play(checkpoint: Path, episodes: int = 5, epsilon: float = 0.01, video_dir: str | None = None, seed: int = 0):
    """Play full games (all lives, unclipped score) and return the scores."""
    ckpt = torch.load(checkpoint, map_location="cpu")
    env_id = ckpt["params"]["env_id"]
    env = make_env(env_id, seed, video_dir, record_all=True, train=False)
    q_network = QNetwork(env.action_space.n)
    q_network.load_state_dict(ckpt["state_dict"])
    q_network.eval()
    rng = np.random.default_rng(seed)

    scores = []
    obs, _ = env.reset(seed=seed)
    while len(scores) < episodes:
        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = int(q_network(torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32)).argmax().item())
        obs, _, terminated, truncated, info = env.step(action)
        if "episode" in info:
            scores.append(float(info["episode"]["r"]))
            print(f"Game {len(scores)}: score {scores[-1]:.0f}")
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    print(f"Mean score over {episodes} games: {np.mean(scores):.1f} (best {max(scores):.0f})")
    return scores


def plot_returns(run_dirs: list[Path], window: int = 100, out: Path = Path("outputs/returns.png")) -> Path:
    """Plot the rolling-average episodic return of one or more runs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for run_dir in run_dirs:
        df = pd.read_csv(Path(run_dir) / "returns.csv")
        rolling = df["episodic_return"].rolling(window, min_periods=1).mean()
        ax.plot(df["global_step"], rolling, label=Path(run_dir).name)
    ax.set_xlabel("Global step")
    ax.set_ylabel(f"Episodic return ({window}-episode rolling mean)")
    ax.set_title("DQN on Breakout")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out
