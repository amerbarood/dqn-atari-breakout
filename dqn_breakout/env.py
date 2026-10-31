"""Atari environment with the standard DQN preprocessing."""

from __future__ import annotations

import ale_py
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

gym.register_envs(ale_py)


def make_env(env_id: str, seed: int, video_dir: str | None = None, record_all: bool = False, train: bool = True):
    """Create the preprocessed environment.

    Pipeline (as in the original notebook): no-op starts, frame skip 4 with max
    pooling, end-of-life as episode end, FIRE on reset, reward clipping,
    84x84 grayscale, stack of 4 frames.

    With train=False, life loss and reward clipping are not applied, so the
    reported return is the real game score.

    Videos use gymnasium's default schedule (episodes 0, 1, 8, 27, ..., then
    every 1000th), like the original run, or every episode with record_all.
    """
    env = gym.make(env_id, render_mode="rgb_array" if video_dir else None)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if video_dir:
        trigger = (lambda ep: True) if record_all else None
        env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=trigger, disable_logger=True)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if train:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if train:
        env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
