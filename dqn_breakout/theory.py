"""Theory task: value iteration on a 3x4 grid world (Q-learning by hand).

Discount 0.7, the intended move succeeds with probability 0.8, horizon 100.
Terminal cells (reward -1 at row 0 col 1 and +1 at row 0 col 3) keep their value.
"""

from __future__ import annotations

import numpy as np

GAMMA = 0.7
P = 0.8
HORIZON = 100


def rewards() -> np.ndarray:
    w = np.zeros((3, 4))
    w[0, 1] = -1
    w[0, 3] = 1
    return w


def transition_kernels(p: float = P) -> np.ndarray:
    """pi[r, c] is a 3x3 kernel: the probability of moving to each neighbour of (r, c)
    under the chosen policy. A 1 in the centre marks a terminal cell."""
    q = 1 - p
    pi = np.zeros((3, 4, 3, 3))
    pi[0, 0] = [[0, 0, 0], [0, 0, q], [0, p, 0]]
    pi[1, 0] = [[0, q, 0], [0, 0, 0], [0, p, 0]]
    pi[2, 0] = [[0, q, 0], [0, 0, p], [0, 0, 0]]
    pi[0, 1] = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    pi[1, 1] = np.zeros((3, 3))  # wall
    pi[2, 1] = [[0, 0, 0], [q, 0, p], [0, 0, 0]]
    pi[0, 2] = [[0, 0, 0], [q / 2, 0, p], [0, q / 2, 0]]
    pi[1, 2] = [[0, q / 2, 0], [0, 0, p], [0, q / 2, 0]]
    pi[2, 2] = [[0, q / 2, 0], [q / 2, 0, p], [0, 0, 0]]
    pi[0, 3] = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    pi[1, 3] = [[0, p, 0], [q / 2, 0, 0], [0, q / 2, 0]]
    pi[2, 3] = [[0, p, 0], [q, 0, 0], [0, 0, 0]]
    return pi


def policy_evaluation(gamma: float = GAMMA, p: float = P, horizon: int = HORIZON) -> np.ndarray:
    w = rewards()
    pi = transition_kernels(p)
    rows, cols = w.shape
    values = np.pad(w.copy(), 1)
    for _ in range(horizon):
        new = np.zeros_like(values)
        for r in range(rows):
            for c in range(cols):
                g = 1.0 if pi[r, c][1, 1] == 1 else gamma
                new[r + 1, c + 1] = g * np.sum(values[r : r + 3, c : c + 3] * pi[r, c])
        values = new
    return np.round(values[1 : rows + 1, 1 : cols + 1], 4)
