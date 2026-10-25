"""Monte Carlo tree search for the two-dimensional HP lattice model."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field

from tqdm import tqdm

Point = tuple[int, int]

# Keep the action order and labels used by gym_lattice's Lattice2DEnv.
ACTIONS: tuple[tuple[str, Point], ...] = (
    ("L", (-1, 0)),
    ("D", (0, 1)),
    ("U", (0, -1)),
    ("R", (1, 0)),
)


@dataclass
class Node:
    coords: tuple[Point, ...]
    parent: Node | None = None
    untried_actions: list[int] = field(default_factory=list)
    children: list[Node] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0


def legal_actions(coords: tuple[Point, ...]) -> list[int]:
    """Return moves from the chain end that do not revisit an occupied cell."""
    occupied = set(coords)
    x, y = coords[-1]
    return [
        action
        for action, (_, (dx, dy)) in enumerate(ACTIONS)
        if (x + dx, y + dy) not in occupied
    ]


def count_contacts(sequence: str, coords: tuple[Point, ...]) -> int:
    """Count non-bonded neighboring H-H pairs in a completed or partial fold."""
    index_at = {point: index for index, point in enumerate(coords)}
    contacts = 0
    for index, (x, y) in enumerate(coords):
        if sequence[index] != "H":
            continue
        for neighbour in ((x + 1, y), (x, y + 1)):
            other = index_at.get(neighbour)
            if other is not None and other > index + 1 and sequence[other] == "H":
                contacts += 1
    return contacts


def rollout(
    sequence: str, coords: tuple[Point, ...], rng: random.Random
) -> tuple[tuple[Point, ...], bool]:
    """Randomly extend a fold until it completes or becomes trapped."""
    while len(coords) < len(sequence):
        options = legal_actions(coords)
        if not options:
            return coords, False
        action = rng.choice(options)
        _, (dx, dy) = ACTIONS[action]
        x, y = coords[-1]
        coords += ((x + dx, y + dy),)
    return coords, True


def search(
    sequence: str,
    iterations: int = 5000,
    exploration: float = math.sqrt(2),
    seed: int | None = None,
) -> tuple[int, tuple[Point, ...]] | None:
    """Run UCT-based MCTS and return the best complete fold found."""
    if not sequence or any(residue not in "HP" for residue in sequence):
        raise ValueError("sequence must contain only H and P residues")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if exploration < 0:
        raise ValueError("exploration must be non-negative")

    rng = random.Random(seed)
    root = Node(coords=((0, 0),))
    root.untried_actions = legal_actions(root.coords)
    best: tuple[int, tuple[Point, ...]] | None = None

    for _ in tqdm(range(iterations), desc="MCTS", unit="iteration"):
        node = root

        # Selection: descend through fully expanded nodes using UCT.
        while len(node.coords) < len(sequence) and not node.untried_actions:
            if not node.children:
                break
            node = max(
                node.children,
                key=lambda child: (
                    child.value / child.visits
                    + exploration
                    * math.sqrt(math.log(node.visits) / child.visits)
                ),
            )

        # Expansion: add one previously unexplored legal move.
        if len(node.coords) < len(sequence) and node.untried_actions:
            action = rng.choice(node.untried_actions)
            node.untried_actions.remove(action)
            _, (dx, dy) = ACTIONS[action]
            x, y = node.coords[-1]
            child = Node(
                coords=node.coords + ((x + dx, y + dy),),
                parent=node,
            )
            child.untried_actions = legal_actions(child.coords)
            node.children.append(child)
            node = child

        # Simulation and backpropagation.
        final_coords, complete = rollout(sequence, node.coords, rng)
        contacts = count_contacts(sequence, final_coords)
        if complete and (best is None or contacts > best[0]):
            best = (contacts, final_coords)

        # All completed folds score in [0, 1]; an incomplete fold scores -1.
        reward = contacts / max(1, len(sequence)) if complete else -1.0
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    return best


def parse_sequence(value: str) -> str:
    sequence = value.strip().upper()
    if not sequence or any(residue not in "HP" for residue in sequence):
        raise argparse.ArgumentTypeError("sequence must contain only H and P")
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search for a low-energy HP protein fold with MCTS."
    )
    parser.add_argument("sequence", type=parse_sequence, help="HP sequence, e.g. HHPHH")
    parser.add_argument(
        "--iterations", type=int, default=5000, help="MCTS simulations (default: 5000)"
    )
    parser.add_argument(
        "--exploration",
        type=float,
        default=math.sqrt(2),
        help="UCT exploration constant (default: sqrt(2))",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()

    try:
        result = search(args.sequence, args.iterations, args.exploration, args.seed)
    except ValueError as error:
        parser.error(str(error))

    if result is None:
        print("No complete fold found; try increasing --iterations.")
        return

    contacts, coords = result
    labels_by_delta = {delta: label for label, delta in ACTIONS}
    moves = "".join(
        labels_by_delta[(end[0] - start[0], end[1] - start[1])]
        for start, end in zip(coords, coords[1:])
    )
    print(f"Sequence: {args.sequence}")
    print(f"Moves: {moves or '(none)'}")
    print(f"H-H contacts: {contacts}")
    print(f"Energy (negative contacts): {-contacts}")
    print(f"Coordinates: {coords}")


if __name__ == "__main__":
    main()
