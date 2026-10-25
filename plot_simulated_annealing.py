"""Run repeated simulated-annealing benchmarks and plot their results."""

from __future__ import annotations

import argparse
import csv
import shutil
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mcts
import simulated_annealing


DEFAULT_BENCHMARK = Path(__file__).with_name("benchmark.txt")
DEFAULT_OUTPUT = Path(__file__).parent / "solver_results/150k_simulated_annealing"
DEFAULT_ITERATIONS = 150_000
DEFAULT_RUNS = 5
PROGRESS_INTERVAL = 1_000


def run_benchmark(
    benchmark_path: Path,
    output_dir: Path,
    iterations: int = DEFAULT_ITERATIONS,
    runs: int = DEFAULT_RUNS,
    first_seed: int = 7,
    progress_interval: int = PROGRESS_INTERVAL,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run each benchmark sequence repeatedly and save raw results/traces."""
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 2:
        raise ValueError("runs must be an integer of at least 2")
    if isinstance(first_seed, bool) or not isinstance(first_seed, int):
        raise ValueError("first_seed must be an integer")
    if (
        isinstance(progress_interval, bool)
        or not isinstance(progress_interval, int)
        or progress_interval < 1
    ):
        raise ValueError("progress_interval must be a positive integer")

    benchmark = mcts.load_benchmark(benchmark_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_benchmark = output_dir / "benchmark.txt"
    if benchmark_path.resolve() != saved_benchmark.resolve():
        shutil.copyfile(benchmark_path, saved_benchmark)
    results: list[dict[str, object]] = []
    progress_rows: list[dict[str, object]] = []

    for name, sequence, reference_energy in benchmark:
        for run_index in range(runs):
            seed = first_seed + run_index
            trace: list[dict[str, object]] = []

            def record_progress(step: int, elapsed: float, best_contacts: int) -> None:
                trace.append(
                    {
                        "name": name,
                        "sequence": sequence,
                        "run": run_index + 1,
                        "seed": seed,
                        "step": step,
                        "elapsed_seconds": elapsed,
                        "best_contacts": best_contacts,
                        "best_energy": -best_contacts,
                    }
                )

            record_progress(0, 0.0, 0)
            start = time.perf_counter()
            contacts, coords = simulated_annealing.search(
                sequence,
                iterations=iterations,
                seed=seed,
                progress_callback=record_progress,
                progress_interval=progress_interval,
            )
            elapsed = time.perf_counter() - start
            time_to_best = min(
                int(point["step"])
                for point in trace
                if int(point["best_contacts"]) == contacts
            )
            results.append(
                {
                    "name": name,
                    "sequence": sequence,
                    "iterations": iterations,
                    "run": run_index + 1,
                    "seed": seed,
                    "contacts": contacts,
                    "energy": -contacts,
                    "complete": len(coords) == len(sequence),
                    "elapsed_seconds": f"{elapsed:.6f}",
                    "time_to_final_best_iterations": time_to_best,
                    "reference_energy": reference_energy,
                    "meets_reference": (
                        -contacts <= reference_energy
                        if reference_energy is not None
                        else ""
                    ),
                }
            )
            progress_rows.extend(
                {**point, "iterations": iterations}
                for point in trace
            )
            print(
                f"{name} run={run_index + 1}/{runs} seed={seed}: "
                f"{iterations} iterations, energy={-contacts}, time={elapsed:.2f}s",
                flush=True,
            )

    results_path = output_dir / "simulated_annealing_results.csv"
    progress_path = output_dir / "simulated_annealing_progress.csv"
    _write_csv(results_path, results)
    _write_csv(progress_path, progress_rows)
    time_plot_path, energy_plot_path = plot_results(
        results, progress_rows, output_dir, progress_interval
    )
    print(f"Saved results to {results_path}")
    print(f"Saved progress to {progress_path}")
    print(f"Saved iteration bar chart to {time_plot_path}")
    print(f"Saved best-so-far energy chart to {energy_plot_path}")
    return results, progress_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_and_sample_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("at least one value is required")
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _sequence_colors(count: int) -> list[tuple[float, float, float, float]]:
    color_map = plt.get_cmap("viridis")
    return [color_map(index / max(1, count - 1)) for index in range(count)]


def plot_results(
    rows: list[dict[str, object]],
    progress_rows: list[dict[str, object]],
    output_dir: Path,
    progress_interval: int = PROGRESS_INTERVAL,
) -> tuple[Path, Path]:
    """Plot mean iteration-to-best with SD bars and raw per-sequence traces."""
    if not rows or not progress_rows:
        raise ValueError("results and progress data must not be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(str(row["name"]) for row in rows))
    colors = _sequence_colors(len(names))

    time_plot_path = output_dir / "simulated_annealing_time_to_best_iterations.png"
    energy_plot_path = output_dir / "simulated_annealing_best_so_far_energy.png"

    figure_height = max(7, 0.62 * len(names))
    figure, axis = plt.subplots(
        figsize=(14, figure_height),
        constrained_layout=True,
    )
    positions = list(range(len(names)))
    for index, name in enumerate(names):
        values = [
            float(row["time_to_final_best_iterations"])
            for row in rows
            if row["name"] == name
        ]
        mean, standard_deviation = _mean_and_sample_sd(values)
        axis.barh(
            index,
            mean,
            xerr=standard_deviation,
            color=colors[index],
            alpha=0.9,
            capsize=4,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )
    axis.set_yticks(positions, names, fontsize=9)
    axis.invert_yaxis()
    axis.set_xlabel("Iterations to first reach final best energy (mean ± SD)")
    axis.set_title("Simulated annealing: iterations to final best energy")
    axis.grid(axis="x", alpha=0.25)
    figure.savefig(time_plot_path, dpi=180)
    plt.close(figure)

    iterations = int(rows[0]["iterations"])
    sample_steps = list(range(0, iterations + 1, progress_interval))
    if not sample_steps or sample_steps[-1] != iterations:
        sample_steps.append(iterations)

    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for index, name in enumerate(names):
        run_traces = []
        for run in sorted(
            {
                int(row["run"])
                for row in progress_rows
                if row["name"] == name
            }
        ):
            trace = sorted(
                (
                    row
                    for row in progress_rows
                    if row["name"] == name and int(row["run"]) == run
                ),
                key=lambda row: int(row["step"]),
            )
            cursor = 0
            energy_at_steps: list[float] = []
            for step in sample_steps:
                while (
                    cursor + 1 < len(trace)
                    and int(trace[cursor + 1]["step"]) <= step
                ):
                    cursor += 1
                energy_at_steps.append(float(trace[cursor]["best_energy"]))
            run_traces.append(energy_at_steps)

        minimum_energy = [
            min(trace[step_index] for trace in run_traces)
            for step_index in range(len(sample_steps))
        ]
        axis.plot(
            sample_steps,
            minimum_energy,
            color=colors[index],
            linewidth=1.8,
            label=name,
        )
    axis.set_xlabel("Solver iterations")
    axis.set_ylabel("Best-so-far energy (negative H-H contacts)")
    axis.set_title(
        f"Simulated annealing: best-so-far energy per sequence "
        f"(minimum of {len({int(row['run']) for row in rows})} runs; unsmoothed)"
    )
    axis.grid(alpha=0.25)
    axis.legend(title="Benchmark sequence", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.savefig(energy_plot_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return time_plot_path, energy_plot_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and plot repeated simulated-annealing benchmark runs."
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--first-seed", type=int, default=7)
    parser.add_argument("--progress-interval", type=int, default=PROGRESS_INTERVAL)
    args = parser.parse_args()
    try:
        run_benchmark(
            args.benchmark,
            args.output_dir,
            iterations=args.iterations,
            runs=args.runs,
            first_seed=args.first_seed,
            progress_interval=args.progress_interval,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
