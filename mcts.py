"""Monte Carlo tree search for the two-dimensional HP lattice model."""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable

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
    iterations: int = 100_000,
    exploration: float = math.sqrt(2),
    seed: int | None = None,
    show_progress: bool = True,
    progress_callback: Callable[[int, float, int], None] | None = None,
    progress_interval: int = 1_000,
) -> tuple[int, tuple[Point, ...]] | None:
    """Run UCT-based MCTS and return the best complete fold found."""
    sequence = sequence.strip().upper()
    if not sequence or any(residue not in "HP" for residue in sequence):
        raise ValueError("sequence must contain only H and P residues")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be at least 1")
    if not math.isfinite(exploration) or exploration < 0:
        raise ValueError("exploration must be finite and non-negative")
    if (
        isinstance(progress_interval, bool)
        or not isinstance(progress_interval, int)
        or progress_interval < 1
    ):
        raise ValueError("progress_interval must be a positive integer")

    rng = random.Random(seed)
    root = Node(coords=((0, 0),))
    root.untried_actions = legal_actions(root.coords)
    best: tuple[int, tuple[Point, ...]] | None = None
    best_contacts = 0
    start_time = time.perf_counter()

    iteration_range = range(iterations)
    if show_progress:
        iteration_range = tqdm(iteration_range, desc="MCTS", unit="iteration")
    for step in iteration_range:
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
        improved = complete and (best is None or contacts > best[0])
        if improved:
            best = (contacts, final_coords)
            best_contacts = contacts

        # All completed folds score in [0, 1]; an incomplete fold scores -1.
        reward = contacts / max(1, len(sequence)) if complete else -1.0
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

        if progress_callback is not None and (
            improved or (step + 1) % progress_interval == 0 or step + 1 == iterations
        ):
            progress_callback(
                step + 1,
                time.perf_counter() - start_time,
                best_contacts,
            )

    return best


def parse_sequence(value: str) -> str:
    sequence = value.strip().upper()
    if not sequence or any(residue not in "HP" for residue in sequence):
        raise argparse.ArgumentTypeError("sequence must contain only H and P")
    return sequence


def coordinates_to_moves(coords: tuple[Point, ...]) -> str:
    """Convert lattice coordinates to the L/D/U/R labels used by gym-lattice."""
    labels_by_delta = {delta: label for label, delta in ACTIONS}
    try:
        return "".join(
            labels_by_delta[(end[0] - start[0], end[1] - start[1])]
            for start, end in zip(coords, coords[1:])
        )
    except KeyError as error:
        raise ValueError("coordinates must form a unit-step lattice walk") from error


def load_benchmark(path: str | Path) -> list[tuple[str, str, int | None]]:
    """Read whitespace-separated length, sequence, and optional energy rows."""
    benchmark_path = Path(path)
    sequences = []
    with benchmark_path.open(encoding="utf-8") as benchmark_file:
        for line_number, line in enumerate(benchmark_file, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError(
                    f"{benchmark_path}:{line_number}: expected length and sequence"
                )
            try:
                declared_length = int(fields[0])
                reference_energy = int(fields[2]) if len(fields) > 2 else None
            except ValueError as error:
                raise ValueError(
                    f"{benchmark_path}:{line_number}: invalid length or energy"
                ) from error
            sequence = fields[1].strip().upper()
            if not sequence or any(residue not in "HP" for residue in sequence):
                raise ValueError(
                    f"{benchmark_path}:{line_number}: sequence must contain only H and P"
                )
            if declared_length != len(sequence):
                raise ValueError(
                    f"{benchmark_path}:{line_number}: declared length does not "
                    "match sequence"
                )
            sequences.append(
                (
                    f"sequence-{len(sequences) + 1:02d}",
                    sequence,
                    reference_energy,
                )
            )
    if not sequences:
        raise ValueError(f"{benchmark_path}: no benchmark sequences found")
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search for a low-energy HP protein fold with MCTS."
    )
    parser.add_argument(
        "sequence",
        nargs="?",
        type=parse_sequence,
        help="HP sequence, e.g. HHPHH (omit when using --benchmark)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100_000,
        help="MCTS simulations (default: 100000, matching simulated annealing)",
    )
    parser.add_argument(
        "--exploration",
        type=float,
        default=math.sqrt(2),
        help="UCT exploration constant (default: sqrt(2))",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--benchmark",
        type=Path,
        help="evaluate sequences from a length/sequence/optional-energy file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write benchmark results to CSV (requires --benchmark)",
    )
    args = parser.parse_args()

    if args.benchmark is not None:
        if args.sequence is not None:
            parser.error("do not provide a sequence together with --benchmark")
        if args.output is None:
            parser.error("--output is required with --benchmark")
    elif args.sequence is None:
        parser.error("a sequence is required unless --benchmark is specified")

    try:
        if args.benchmark is not None:
            rows = []
            for name, sequence, reference_energy in load_benchmark(args.benchmark):
                start = time.perf_counter()
                result = search(
                    sequence,
                    args.iterations,
                    args.exploration,
                    args.seed,
                    show_progress=False,
                )
                elapsed = time.perf_counter() - start
                contacts, coords = result if result is not None else (0, ())
                energy = -contacts if result is not None else None
                rows.append(
                    {
                        "algorithm": "mcts",
                        "name": name,
                        "sequence": sequence,
                        "iterations": args.iterations,
                        "seed": args.seed,
                        "contacts": contacts,
                        "energy": energy,
                        "complete": result is not None,
                        "elapsed_seconds": f"{elapsed:.6f}",
                        "reference_energy": reference_energy,
                        "meets_reference": (
                            energy <= reference_energy
                            if energy is not None and reference_energy is not None
                            else ""
                        ),
                        "moves": coordinates_to_moves(coords) if coords else "",
                    }
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved {len(rows)} benchmark results to {args.output}")
            return

        sequence = args.sequence
        start = time.perf_counter()
        result = search(
            sequence, args.iterations, args.exploration, args.seed
        )
        elapsed = time.perf_counter() - start
    except ValueError as error:
        parser.error(str(error))

    if result is None:
        print("No complete fold found; try increasing --iterations.")
        return

    contacts, coords = result
    moves = coordinates_to_moves(coords)
    print(f"Sequence: {sequence}")
    print(f"Iterations: {args.iterations}")
    print(f"Seed: {args.seed}")
    print(f"Moves: {moves or '(none)'}")
    print(f"H-H contacts: {contacts}")
    print(f"Energy (negative contacts): {-contacts}")
    print(f"Coordinates: {coords}")
    print(f"Elapsed seconds: {elapsed:.6f}")


if __name__ == "__main__":
    main()
