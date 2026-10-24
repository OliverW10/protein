"""Control the lattice environment interactively from a terminal."""

import argparse
import curses
import re
from typing import Any

from gym_lattice.envs import Lattice2DEnv


ACTION_BY_KEY = {
    curses.KEY_LEFT: 0,
    curses.KEY_DOWN: 1,
    curses.KEY_UP: 2,
    curses.KEY_RIGHT: 3,
}
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def parse_sequence(value: str) -> str:
    sequence = value.upper()
    if not sequence or not set(sequence) <= set("HP"):
        raise argparse.ArgumentTypeError("sequence must contain one or more H or P residues")
    return sequence


def draw(screen: Any, env: Lattice2DEnv, status: str) -> None:
    """Draw the current environment state and controls."""
    height, width = screen.getmaxyx()
    screen.erase()

    lines = [
        f"Sequence: {env.seq}  ({len(env.state)}/{len(env.seq)} placed)",
        "Use arrow keys to place the next residue. Press r to restart or q to quit.",
        "",
    ]
    board = env.render(mode="ansi").getvalue()
    lines.extend(ANSI_ESCAPE.sub("", board).splitlines())

    if height > 1 and width > 1:
        screen.addnstr(height - 1, 0, status, width - 1)
        for row, line in enumerate(lines[: height - 1]):
            screen.addnstr(row, 0, line, width - 1)
    screen.refresh()


def control(screen: Any, env: Lattice2DEnv) -> None:
    """Run the arrow-key control loop."""
    screen.keypad(True)
    status = "Ready."

    while True:
        draw(screen, env, status)
        key = screen.getch()

        if key in (ord("q"), ord("Q")):
            return
        if key in (ord("r"), ord("R")):
            env.reset()
            status = "Restarted."
            continue
        if key not in ACTION_BY_KEY:
            continue
        if env.done:
            status = "Episode complete. Press r to restart or q to quit."
            continue

        action = ACTION_BY_KEY[key]
        _, reward, terminated, _, info = env.step(action)
        status = (
            f"Reward: {reward:.2f} | Collisions: {info['collisions']} | "
            f"Actions: {' '.join(info['actions'])}"
        )
        if terminated:
            status = f"Episode complete. {status} Press r to restart or q to quit."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Place residues in the lattice using the terminal arrow keys."
    )
    parser.add_argument(
        "--sequence",
        type=parse_sequence,
        default="HPHPH",
        help="H/P sequence to fold (default: HPHPH)",
    )
    args = parser.parse_args()

    env = Lattice2DEnv(len(args.sequence), args.sequence)
    curses.wrapper(control, env)


if __name__ == "__main__":
    main()
