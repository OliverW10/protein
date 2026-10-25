"""Run MCTS and simulated annealing on identical HP benchmark inputs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mcts
import simulated_annealing


DEFAULT_BENCHMARK = Path(__file__).with_name("benchmark.txt")
DEFAULT_OUTPUT = Path("logs/solver_comparison.csv")
PROGRESS_INTERVAL = 1_000
SMOOTHING_WINDOW = 7


def compare(
    benchmark_path: Path,
    output_path: Path,
    iterations: int,
    seeds: list[int],
    exploration: float,
    progress_interval: int = PROGRESS_INTERVAL,
) -> list[dict[str, object]]:
    """Run both solvers and persist final scores plus best-so-far traces."""
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not seeds:
        raise ValueError("at least one seed is required")
    if not math.isfinite(exploration) or exploration < 0:
        raise ValueError("exploration must be finite and non-negative")
    if (
        isinstance(progress_interval, bool)
        or not isinstance(progress_interval, int)
        or progress_interval < 1
    ):
        raise ValueError("progress_interval must be a positive integer")
    benchmark = mcts.load_benchmark(benchmark_path)
    rows: list[dict[str, object]] = []
    progress_rows: list[dict[str, object]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_name(f"{output_path.stem}_progress.csv")

    for name, sequence, reference_energy in benchmark:
        for seed in seeds:
            for algorithm in ("mcts", "simulated_annealing"):
                trace: list[dict[str, object]] = []

                def record_progress(
                    step: int, elapsed: float, best_contacts: int
                ) -> None:
                    trace.append(
                        {
                            "algorithm": algorithm,
                            "name": name,
                            "sequence": sequence,
                            "seed": seed,
                            "step": step,
                            "elapsed_seconds": elapsed,
                            "best_contacts": best_contacts,
                            "best_energy": -best_contacts,
                        }
                    )

                record_progress(0, 0.0, 0)
                start = time.perf_counter()
                if algorithm == "mcts":
                    result = mcts.search(
                        sequence,
                        iterations=iterations,
                        exploration=exploration,
                        seed=seed,
                        show_progress=False,
                        progress_callback=record_progress,
                        progress_interval=progress_interval,
                    )
                else:
                    result = simulated_annealing.search(
                        sequence,
                        iterations=iterations,
                        seed=seed,
                        progress_callback=record_progress,
                        progress_interval=progress_interval,
                    )
                elapsed = time.perf_counter() - start

                if result is None:
                    contacts = 0
                    coords = ()
                    energy = None
                    complete = False
                    moves = ""
                else:
                    contacts, coords = result
                    energy = -contacts
                    complete = len(coords) == len(sequence)
                    moves = simulated_annealing.coordinates_to_moves(coords)
                best_step_time = next(
                    (
                        float(point["elapsed_seconds"])
                        for point in trace
                        if int(point["best_contacts"]) == contacts
                    ),
                    elapsed,
                )
                progress_rows.extend(
                    {
                        **point,
                        "iterations": iterations,
                    }
                    for point in trace
                )

                rows.append(
                    {
                        "algorithm": algorithm,
                        "name": name,
                        "sequence": sequence,
                        "iterations": iterations,
                        "seed": seed,
                        "contacts": contacts,
                        "energy": energy,
                        "complete": complete,
                        "elapsed_seconds": f"{elapsed:.6f}",
                        "time_to_final_best_seconds": f"{best_step_time:.6f}",
                        "reference_energy": reference_energy,
                        "meets_reference": (
                            energy <= reference_energy
                            if energy is not None and reference_energy is not None
                            else ""
                        ),
                        "moves": moves,
                    }
                )
                print(
                    f"{name} {algorithm} seed={seed}: {iterations} iterations, "
                    f"energy={energy}, time={elapsed:.2f}s",
                    flush=True,
                )
                _write_csv(output_path, rows)
                _write_csv(progress_path, progress_rows)

    plot_results(rows, progress_rows, output_path, progress_interval)
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _moving_average(values: list[float], window: int) -> list[float]:
    """Return a centered moving average with truncated edges."""
    smoothed = []
    radius = window // 2
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed.append(statistics.fmean(values[start:end]))
    return smoothed


def plot_results(
    rows: list[dict[str, object]],
    progress_rows: list[dict[str, object]],
    output_path: Path,
    progress_interval: int = PROGRESS_INTERVAL,
) -> tuple[Path, Path]:
    """Plot time to final best energy and smoothed mean best-so-far energy."""
    algorithms = ("mcts", "simulated_annealing")
    names = list(dict.fromkeys(str(row["name"]) for row in rows))
    colors = {"mcts": "#4477aa", "simulated_annealing": "#cc6677"}
    time_plot_path = output_path.with_name(f"{output_path.stem}_time_to_best.png")
    progress_plot_path = output_path.with_name(f"{output_path.stem}_smoothed_energy.png")

    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    offsets = {"mcts": -0.12, "simulated_annealing": 0.12}
    for algorithm in algorithms:
        algo_rows = [row for row in rows if row["algorithm"] == algorithm]
        positions = [
            names.index(str(row["name"])) + offsets[algorithm]
            for row in algo_rows
        ]
        times = [float(row["time_to_final_best_seconds"]) for row in algo_rows]
        axis.scatter(
            times,
            positions,
            label=algorithm.replace("_", " ").title(),
            color=colors[algorithm],
            s=36,
            alpha=0.85,
        )
    axis.set_yticks(range(len(names)), names)
    axis.invert_yaxis()
    axis.set_xlabel("Elapsed wall time to first reach final best energy (seconds)")
    axis.set_title(
        f"Time to final best energy ({rows[0]['iterations']:,} iterations per run)"
    )
    axis.grid(axis="x", alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(time_plot_path, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    sample_steps = list(range(0, int(rows[0]["iterations"]) + 1, progress_interval))
    if sample_steps[-1] != int(rows[0]["iterations"]):
        sample_steps.append(int(rows[0]["iterations"]))
    for algorithm in algorithms:
        curves = []
        for name in names:
            matching = sorted(
                (
                    row
                    for row in progress_rows
                    if row["algorithm"] == algorithm and row["name"] == name
                ),
                key=lambda row: int(row["step"]),
            )
            if not matching:
                continue
            cursor = 0
            best_energy_curve = []
            for step in sample_steps:
                while (
                    cursor + 1 < len(matching)
                    and int(matching[cursor + 1]["step"]) <= step
                ):
                    cursor += 1
                best_energy_curve.append(float(matching[cursor]["best_energy"]))
            curves.append(best_energy_curve)
        if curves:
            mean_energy = [
                statistics.fmean(curve[index] for curve in curves)
                for index in range(len(sample_steps))
            ]
            axis.plot(
                sample_steps,
                _moving_average(mean_energy, SMOOTHING_WINDOW),
                label=algorithm.replace("_", " ").title(),
                color=colors[algorithm],
                linewidth=2,
            )
    axis.set_xlabel("Solver iterations")
    axis.set_ylabel("Mean best-so-far energy (negative H-H contacts)")
    axis.set_title(
        f"Smoothed best-so-far energy across {len(names)} benchmark sequences"
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(progress_plot_path, dpi=180)
    plt.close(figure)
    return time_plot_path, progress_plot_path


def print_summary(rows: list[dict[str, object]]) -> None:
    """Print average contacts, completion, reference hits, and runtime."""
    print(
        "Algorithm, runs, mean contacts, completion rate, reference hit rate, "
        "mean seconds"
    )
    for algorithm in ("mcts", "simulated_annealing"):
        model_rows = [row for row in rows if row["algorithm"] == algorithm]
        if not model_rows:
            continue
        count = len(model_rows)
        complete = sum(bool(row["complete"]) for row in model_rows)
        contacts = sum(int(row["contacts"]) for row in model_rows) / count
        runtimes = sum(float(row["elapsed_seconds"]) for row in model_rows) / count
        time_to_best = sum(
            float(row["time_to_final_best_seconds"]) for row in model_rows
        ) / count
        reference_rows = [
            row for row in model_rows if row["meets_reference"] != ""
        ]
        reference_hits = (
            sum(bool(row["meets_reference"]) for row in reference_rows)
            / len(reference_rows)
            if reference_rows
            else 0.0
        )
        print(
            f"{algorithm}, {count}, {contacts:.3f}, {complete / count:.1%}, "
            f"{reference_hits:.1%}, {runtimes:.6f} "
            f"(time-to-best mean {time_to_best:.6f}s)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MCTS and simulated annealing on the same HP sequences."
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 14, 21])
    parser.add_argument("--exploration", type=float, default=2**0.5)
    parser.add_argument("--progress-interval", type=int, default=PROGRESS_INTERVAL)
    args = parser.parse_args()

    try:
        rows = compare(
            args.benchmark,
            args.output,
            args.iterations,
            args.seeds,
            args.exploration,
            args.progress_interval,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print_summary(rows)
    print(f"Saved per-sequence results to {args.output}")
    print(
        f"Saved progress to {args.output.with_name(args.output.stem + '_progress.csv')}"
    )
    print(
        f"Saved time-to-best plot to "
        f"{args.output.with_name(args.output.stem + '_time_to_best.png')}"
    )
    print(
        f"Saved smoothed energy plot to "
        f"{args.output.with_name(args.output.stem + '_smoothed_energy.png')}"
    )


if __name__ == "__main__":
    main()
