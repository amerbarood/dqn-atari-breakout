"""Command-line entry point.

Examples:
    python main.py train --run F                          # most stable configuration
    python main.py train --run A --total-timesteps 200000 # shorter run
    python main.py play runs/<run>/model.pt --episodes 5 --video videos/eval
    python main.py plot runs/<run_A> runs/<run_F>
    python main.py theory                                 # grid-world value iteration
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dqn_breakout.config import RUNS, get_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deep Q-learning on Atari Breakout")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train a DQN agent with one of the run configurations")
    p.add_argument("--run", choices=sorted(RUNS), default="F", help="hyperparameter set (see dqn_breakout/config.py)")
    p.add_argument("--total-timesteps", type=int)
    p.add_argument("--buffer-size", type=int)
    p.add_argument("--learning-starts", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--gamma", type=float)
    p.add_argument("--exploration-fraction", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--no-video", action="store_true", help="don't record gameplay videos")
    p.add_argument("--checkpoint-every", type=int, default=250_000)

    p = sub.add_parser("play", help="play full games with a trained agent")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--video", default=None, help="folder to save gameplay videos")

    p = sub.add_parser("plot", help="plot rolling episodic return for one or more runs")
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--out", type=Path, default=Path("outputs/returns.png"))

    sub.add_parser("theory", help="value iteration for the grid-world theory task")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train":
        from dqn_breakout.train import train

        params = get_run(
            args.run,
            total_timesteps=args.total_timesteps,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            gamma=args.gamma,
            exploration_fraction=args.exploration_fraction,
            seed=args.seed,
            capture_video=False if args.no_video else None,
        )
        train(params, checkpoint_every=args.checkpoint_every)

    elif args.command == "play":
        from dqn_breakout.evaluate import play

        play(args.checkpoint, args.episodes, args.epsilon, args.video)

    elif args.command == "plot":
        from dqn_breakout.evaluate import plot_returns

        plot_returns(args.run_dirs, args.window, args.out)

    elif args.command == "theory":
        from dqn_breakout.theory import policy_evaluation

        print("State values after value iteration (3x4 grid):")
        print(policy_evaluation())


if __name__ == "__main__":
    main()
