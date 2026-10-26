"""Simulated annealing solver for the two-dimensional HP lattice model."""

from __future__ import annotations

import argparse
import math
import random
import time
from collections.abc import Callable, Sequence

Point = tuple[int, int]
Coordinates = tuple[Point, ...]

SYMMETRIES: tuple[tuple[int, int, int, int], ...] = (
    (1, 0, 0, 1),
    (-1, 0, 0, 1),
    (1, 0, 0, -1),
    (-1, 0, 0, -1),
    (0, 1, 1, 0),
    (0, -1, 1, 0),
    (0, 1, -1, 0),
    (0, -1, -1, 0),
)


def validate_sequence(sequence: str) -> str:
    """Return a normalized HP sequence or raise ``ValueError``."""
    normalized = sequence.strip().upper()
    if not normalized or any(residue not in "HP" for residue in normalized):
        raise ValueError("sequence must contain one or more H/P residues")
    return normalized


def count_contacts(sequence: str, coords: Sequence[Point]) -> int:
    """Count non-bonded neighboring H-H pairs in a self-avoiding fold."""
    index_at = {point: index for index, point in enumerate(coords)}
    contacts = 0
    for index, (x, y) in enumerate(coords):
        if sequence[index] != "H":
            continue
        for neighbour in ((x + 1, y), (x, y + 1)):
            other = index_at.get(neighbour)
            if (
                other is not None
                and other > index + 1
                and sequence[other] == "H"
            ):
                contacts += 1
    return contacts


def _is_self_avoiding(coords: Coordinates) -> bool:
    return len(set(coords)) == len(coords)


def _end_move(coords: Coordinates, rng: random.Random) -> Coordinates:
    """Propose moving the terminal residue to another neighboring lattice site."""
    if len(coords) < 2:
        return coords
    previous_x, previous_y = coords[-2]
    current_x, current_y = coords[-1]
    options = [
        (previous_x + dx, previous_y + dy)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
        if (previous_x + dx, previous_y + dy) != (current_x, current_y)
    ]
    candidate = rng.choice(options)
    if candidate in coords[:-1]:
        return coords
    return coords[:-1] + (candidate,)


def _corner_flip(coords: Coordinates, rng: random.Random) -> Coordinates:
    """Propose flipping one corner residue to the opposite plaquette corner."""
    if len(coords) < 3:
        return coords
    index = rng.randrange(1, len(coords) - 1)
    previous = coords[index - 1]
    current = coords[index]
    following = coords[index + 1]
    candidate = (
        previous[0] + following[0] - current[0],
        previous[1] + following[1] - current[1],
    )
    if candidate in coords[:index] or candidate in coords[index + 1 :]:
        return coords
    return coords[:index] + (candidate,) + coords[index + 1 :]


def _pivot_move(coords: Coordinates, rng: random.Random) -> Coordinates:
    """Rotate or reflect a suffix around a randomly selected pivot residue."""
    if len(coords) < 2:
        return coords
    pivot_index = rng.randrange(len(coords) - 1)
    a, b, c, d = rng.choice(SYMMETRIES[1:])
    pivot_x, pivot_y = coords[pivot_index]
    prefix = coords[: pivot_index + 1]
    suffix = tuple(
        (
            pivot_x + a * (x - pivot_x) + b * (y - pivot_y),
            pivot_y + c * (x - pivot_x) + d * (y - pivot_y),
        )
        for x, y in coords[pivot_index + 1 :]
    )
    candidate = prefix + suffix
    if not _is_self_avoiding(candidate):
        return coords
    return candidate


def propose_move(coords: Coordinates, rng: random.Random) -> Coordinates:
    """Choose and apply one reversible self-avoiding-walk proposal."""
    move = rng.randrange(3)
    if move == 0:
        return _end_move(coords, rng)
    if move == 1:
        return _corner_flip(coords, rng)
    return _pivot_move(coords, rng)


def search(
    sequence: str,
    iterations: int = 100_000,
    initial_temperature: float = 2.0,
    final_temperature: float = 0.01,
    seed: int | None = None,
    progress_callback: Callable[[int, float, int], None] | None = None,
    progress_interval: int = 1_000,
    time_budget_seconds: float | None = None,
) -> tuple[int, Coordinates]:
    """Return the best contact count and fold found by simulated annealing."""
    sequence = validate_sequence(sequence)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if (
        isinstance(progress_interval, bool)
        or not isinstance(progress_interval, int)
        or progress_interval < 1
    ):
        raise ValueError("progress_interval must be a positive integer")
    if (
        not math.isfinite(initial_temperature)
        or not math.isfinite(final_temperature)
        or initial_temperature <= 0
        or final_temperature <= 0
        or final_temperature > initial_temperature
    ):
        raise ValueError(
            "temperatures must be positive and final_temperature "
            "must not exceed initial_temperature"
        )
    if time_budget_seconds is not None and (
        not math.isfinite(time_budget_seconds) or time_budget_seconds <= 0
    ):
        raise ValueError("time_budget_seconds must be finite and positive")

    start_time = time.perf_counter()
    deadline = (
        start_time + time_budget_seconds
        if time_budget_seconds is not None
        else None
    )
    rng = random.Random(seed)
    current: Coordinates = tuple((index, 0) for index in range(len(sequence)))
    current_score = count_contacts(sequence, current)
    best = current
    best_score = current_score
    best_improved = False

    for step in range(iterations):
        elapsed = time.perf_counter() - start_time
        if deadline is not None and elapsed >= time_budget_seconds:
            break
        best_improved = False
        candidate = propose_move(current, rng)
        if candidate != current:
            candidate_score = count_contacts(sequence, candidate)
            progress = (
                min(1.0, elapsed / time_budget_seconds)
                if time_budget_seconds is not None
                else step / max(1, iterations - 1)
            )
            temperature = initial_temperature * (
                final_temperature / initial_temperature
            ) ** progress
            score_delta = candidate_score - current_score

            if score_delta >= 0 or rng.random() < math.exp(score_delta / temperature):
                current = candidate
                current_score = candidate_score
                if candidate_score > best_score:
                    best = candidate
                    best_score = candidate_score
                    best_improved = True

        if progress_callback is not None and (
            best_improved or (step + 1) % progress_interval == 0 or step + 1 == iterations
        ):
            progress_callback(
                step + 1,
                time.perf_counter() - start_time,
                best_score,
            )

    return best_score, best


def coordinates_to_moves(coords: Sequence[Point]) -> str:
    """Convert lattice coordinates to the L/D/U/R labels used by gym-lattice."""
    labels = {
        (-1, 0): "L",
        (0, 1): "D",
        (0, -1): "U",
        (1, 0): "R",
    }
    moves = []
    for start, end in zip(coords, coords[1:]):
        delta = (end[0] - start[0], end[1] - start[1])
        try:
            moves.append(labels[delta])
        except KeyError as error:
            raise ValueError("coordinates must form a unit-step lattice walk") from error
    return "".join(moves)


def parse_sequence(value: str) -> str:
    try:
        return validate_sequence(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find a low-energy HP fold with simulated annealing."
    )
    parser.add_argument("sequence", type=parse_sequence, help="HP sequence, e.g. HHPHH")
    parser.add_argument(
        "--iterations",
        type=int,
        default=100_000,
        help="number of annealing proposals (default: 100000)",
    )
    parser.add_argument(
        "--initial-temperature",
        type=float,
        default=2.0,
        help="starting temperature (default: 2.0)",
    )
    parser.add_argument(
        "--final-temperature",
        type=float,
        default=0.01,
        help="ending temperature (default: 0.01)",
    )
    parser.add_argument(
        "--time-budget-ms",
        type=float,
        help="maximum annealing time in milliseconds (stops before the iteration limit)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()

    try:
        start_time = time.perf_counter()
        contacts, coords = search(
            args.sequence,
            iterations=args.iterations,
            initial_temperature=args.initial_temperature,
            final_temperature=args.final_temperature,
            time_budget_seconds=(
                args.time_budget_ms / 1_000
                if args.time_budget_ms is not None
                else None
            ),
            seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))

    elapsed = time.perf_counter() - start_time
    print(f"Sequence: {args.sequence}")
    print(f"Moves: {coordinates_to_moves(coords) or '(none)'}")
    print(f"H-H contacts: {contacts}")
    print(f"Energy (negative contacts): {-contacts}")
    print(f"Coordinates: {coords}")
    if args.time_budget_ms is not None:
        print(f"Elapsed time: {elapsed * 1_000:.3f} ms")


if __name__ == "__main__":
    main()
