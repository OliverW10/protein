"""Generate charts and a fold image from the latest combined training run."""

from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from stable_baselines3 import DQN, PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from Combined.train import _make_env
from Oliver.envs import count_contacts, get_lattice_env


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "logs" / "combined"
SA_RESULTS = ROOT / "solver_results" / "100k_simulated_annealing" / (
    "simulated_annealing_results.csv"
)
OUTPUT_DIR = ROOT / "Combined" / "results" / "latest_run"
MODEL_PAIRS = [
    ("fc", "dqn", "FC-DQN", "#4477aa"),
    ("fc", "ppo", "FC-PPO", "#66ccee"),
    ("lstm", "dqn", "LSTM-DQN", "#228833"),
    ("lstm", "ppo", "LSTM-PPO", "#ccbb44"),
    ("cnn", "dqn", "CNN-DQN", "#aa3377"),
    ("cnn", "ppo", "CNN-PPO", "#bbbbbb"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def load_latest_summary_rows() -> list[dict[str, str]]:
    rows = []
    for network, algorithm, _, _ in MODEL_PAIRS:
        run_dir = RUN_DIR / f"{network}-{algorithm}" / "seed7"
        settings_path = run_dir / "design_settings.json"
        if not settings_path.exists():
            raise FileNotFoundError(f"Missing completed run metadata: {settings_path}")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if settings["network"] != network or settings["algorithm"] != algorithm:
            raise ValueError(f"Run metadata does not match expected pair: {settings_path}")
        rows.append(
            {
                "network": network,
                "algorithm": algorithm,
                "seed": str(settings["seed"]),
                "timesteps": str(settings["timesteps"]),
                "mean_contacts": str(settings["mean_contacts"]),
                "completion_rate": str(settings["completion_rate"]),
            }
        )
    return rows


def load_latest_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    event_files = list(run_dir.glob("**/events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {run_dir}")
    latest_event = max(event_files, key=lambda path: path.stat().st_mtime_ns)
    accumulator = EventAccumulator(str(latest_event), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]
    result = {}
    for tag in ("rollout/ep_rew_mean", "eval/score"):
        if tag not in tags:
            raise ValueError(f"TensorBoard tag {tag!r} missing from {latest_event}")
        result[tag] = [
            (event.step, event.value) for event in accumulator.Scalars(tag)
        ]
    return result


def plot_training_curves(output_path: Path) -> None:
    figure, reward_axis = plt.subplots(figsize=(13, 7), layout="constrained")
    eval_axis = reward_axis.twinx()

    model_handles = []
    for network, algorithm, label, color in MODEL_PAIRS:
        run_dir = RUN_DIR / f"{network}-{algorithm}" / "seed7"
        scalars = load_latest_scalars(run_dir)
        rewards = sorted(scalars["rollout/ep_rew_mean"])
        eval_scores = sorted(scalars["eval/score"])
        if not rewards or not eval_scores:
            raise ValueError(f"Training or eval curve is empty for {label}")

        if not (run_dir / "eval_history.csv").exists():
            raise FileNotFoundError(f"Missing evaluation history for {label}")

        reward_axis.plot(
            [step for step, _ in rewards],
            [reward for _, reward in rewards],
            color=color,
            linewidth=1.8,
            alpha=0.85,
        )
        eval_axis.plot(
            [step for step, _ in eval_scores],
            [score for _, score in eval_scores],
            color=color,
            linestyle="--",
            marker="o",
            markersize=5,
            linewidth=1.8,
        )
        model_handles.append(Line2D([0], [0], color=color, lw=2, label=label))

    reward_axis.set_xlabel("Environment timesteps")
    reward_axis.set_ylabel("Training episode reward mean")
    eval_axis.set_ylabel("Validation mean H-H contacts")
    reward_axis.set_title(
        "Latest training run: episode reward (solid) with validation eval (dashed)"
    )
    reward_axis.grid(alpha=0.25)
    reward_axis.set_xlim(left=0)
    eval_axis.set_ylim(bottom=0)
    reward_axis.legend(
        handles=model_handles, title="Model pair", loc="upper left", ncol=2
    )
    eval_axis.legend(
        handles=[
            Line2D([0], [0], color="#333333", lw=2, label="Training reward"),
            Line2D(
                [0],
                [0],
                color="#333333",
                lw=2,
                linestyle="--",
                marker="o",
                label="Validation eval",
            ),
        ],
        loc="upper right",
        frameon=True,
    )
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_final_comparison(
    summary_rows: list[dict[str, str]], output_path: Path
) -> None:
    rows_by_pair = {
        (row["network"], row["algorithm"]): row for row in summary_rows
    }
    if len(rows_by_pair) != len(MODEL_PAIRS):
        raise ValueError("Expected one final result row for each of the six pairs")

    labels = [label for _, _, label, _ in MODEL_PAIRS]
    colors = [color for _, _, _, color in MODEL_PAIRS]
    contacts = [
        float(rows_by_pair[(network, algorithm)]["mean_contacts"])
        for network, algorithm, _, _ in MODEL_PAIRS
    ]
    completion = [
        100 * float(rows_by_pair[(network, algorithm)]["completion_rate"])
        for network, algorithm, _, _ in MODEL_PAIRS
    ]

    figure, (contacts_axis, completion_axis) = plt.subplots(
        1, 2, figsize=(13, 5.5), layout="constrained"
    )
    for axis, values, title, ylabel, fmt, pad in (
        (
            contacts_axis,
            contacts,
            "Held-out test contacts",
            "Mean H-H contacts (100 random sequences)",
            "%.2f",
            0.08,
        ),
        (
            completion_axis,
            completion,
            "Completed folds",
            "Completed episodes (%)",
            "%.0f%%",
            8,
        ),
    ):
        bars = axis.bar(labels, values, color=colors)
        axis.bar_label(bars, fmt=fmt, padding=3)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, max(values, default=0) + pad)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    figure.suptitle("Final held-out results by model-algorithm pair")
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def benchmark_20mer_rows(
    test_rows: list[dict[str, str]], benchmark_path: Path
) -> list[tuple[str, str]]:
    benchmark_sequences = {}
    for line in benchmark_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].isdigit() and int(fields[0]) == 20:
            benchmark_sequences[fields[1].upper()] = fields[1].upper()

    fixed_rows = [
        row
        for row in test_rows
        if row["name"].startswith("20mer-") and len(row["sequence"]) == 20
    ]
    selected = []
    for row in fixed_rows:
        sequence = row["sequence"].upper()
        if sequence in benchmark_sequences:
            selected.append((row["name"], sequence))
    if len(selected) != 2:
        raise ValueError(
            f"Expected both fixed 20-mer benchmark sequences, found {len(selected)}"
        )
    return selected


def infer_solution(model, network: str, sequence: str, repeats: int = 20):
    env = _make_env(sequence, network)
    timings = []
    final_info = None
    for _ in range(repeats + 1):
        observation, _ = env.reset()
        start = None
        while True:
            if start is None:
                start = time.perf_counter()
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                elapsed = time.perf_counter() - start
                final_info = info
                break
        if len(timings) < repeats:
            timings.append(elapsed)

    lattice = get_lattice_env(env)
    coords = list(lattice.state)
    if final_info is None or not final_info["complete"]:
        raise ValueError(f"Best agent did not complete benchmark sequence {sequence}")
    contacts = count_contacts(coords, sequence)
    if contacts != final_info["contacts"]:
        raise ValueError("Fold contact count differs from environment terminal metric")
    return {
        "coordinates": coords,
        "contacts": contacts,
        "energy": -contacts,
        "time_mean": statistics.fmean(timings),
        "time_sd": statistics.stdev(timings) if len(timings) > 1 else 0.0,
    }


def plot_solution(
    name: str,
    model_name: str,
    sequence: str,
    result: dict[str, object],
    output_path: Path,
) -> None:
    coords = result["coordinates"]
    index_at = {position: index for index, position in enumerate(coords)}
    figure, axis = plt.subplots(figsize=(9, 8), layout="constrained")

    for first, second in zip(coords, coords[1:]):
        axis.plot(
            [first[0], second[0]],
            [first[1], second[1]],
            color="#8a8a8a",
            linewidth=2,
            zorder=1,
        )
    for index, (x, y) in enumerate(coords):
        if sequence[index] != "H":
            continue
        for neighbor in ((x + 1, y), (x, y + 1)):
            other_index = index_at.get(neighbor)
            if (
                other_index is not None
                and sequence[other_index] == "H"
                and abs(index - other_index) > 1
            ):
                axis.plot(
                    [x, neighbor[0]],
                    [y, neighbor[1]],
                    color="#cc3311",
                    linestyle=":",
                    linewidth=2.5,
                    zorder=2,
                )

    for index, ((x, y), residue) in enumerate(zip(coords, sequence)):
        edge_color = "#228833" if index == 0 else "#cc6677" if index == 19 else "#222222"
        axis.scatter(
            x,
            y,
            s=540,
            c="#222222" if residue == "H" else "white",
            edgecolors=edge_color,
            linewidths=2.2 if index in (0, 19) else 1.2,
            zorder=3,
        )
        axis.text(
            x,
            y,
            f"{index + 1}\n{residue}",
            color="white" if residue == "H" else "#222222",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            zorder=4,
        )

    axis.set_aspect("equal")
    axis.set_xticks(range(min(x for x, _ in coords) - 1, max(x for x, _ in coords) + 2))
    axis.set_yticks(range(min(y for _, y in coords) - 1, max(y for _, y in coords) + 2))
    axis.grid(alpha=0.15)
    axis.set_axisbelow(True)
    axis.set_title(
        f"{name}: {model_name} best checkpoint\n"
        f"Energy {result['energy']} ({result['contacts']} H-H contacts)"
    )
    axis.legend(
        handles=[
            Line2D([0], [0], color="#8a8a8a", lw=2, label="Chain bond"),
            Line2D(
                [0],
                [0],
                color="#cc3311",
                lw=2,
                linestyle=":",
                label="Non-bonded H-H contact",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor="#228833",
                label="Start",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor="#cc6677",
                label="End",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_agent_vs_annealing(
    model_name: str,
    sequences: list[tuple[str, str]],
    agent_results: list[dict[str, object]],
    sa_results: list[dict[str, str]],
    output_path: Path,
) -> None:
    grouped_sa = []
    for _, sequence in sequences:
        rows = [
            row
            for row in sa_results
            if row["sequence"].upper() == sequence
            and int(row["iterations"]) == 100_000
        ]
        if not rows:
            raise ValueError(f"No 100k simulated-annealing results for {sequence}")
        grouped_sa.append(rows)

    figure, (energy_axis, time_axis) = plt.subplots(
        2, 1, figsize=(11, 9), layout="constrained"
    )
    x = list(range(len(sequences)))
    width = 0.34
    agent_color = "#4477aa"
    sa_color = "#cc6677"
    sa_energy_means = [
        statistics.fmean(float(row["energy"]) for row in rows)
        for rows in grouped_sa
    ]
    sa_energy_sds = [
        statistics.stdev(float(row["energy"]) for row in rows)
        if len(rows) > 1
        else 0.0
        for rows in grouped_sa
    ]
    agent_energies = [float(result["energy"]) for result in agent_results]
    agent_energy_bars = energy_axis.bar(
        [position - width / 2 for position in x],
        agent_energies,
        width,
        color=agent_color,
        label=model_name,
    )
    sa_energy_bars = energy_axis.bar(
        [position + width / 2 for position in x],
        sa_energy_means,
        width,
        yerr=sa_energy_sds,
        capsize=5,
        color=sa_color,
        label="Simulated annealing (mean ± SD, 5 runs)",
    )
    energy_axis.bar_label(
        agent_energy_bars,
        labels=[f"{value:.1f}" for value in agent_energies],
        padding=3,
        fontsize=9,
    )
    energy_axis.bar_label(
        sa_energy_bars,
        labels=[f"{value:.1f}" for value in sa_energy_means],
        padding=3,
        fontsize=9,
    )
    for index, rows in enumerate(grouped_sa):
        energy_axis.scatter(
            [x[index] + width / 2] * len(rows),
            [float(row["energy"]) for row in rows],
            color="#7a263a",
            s=24,
            zorder=3,
        )
    energy_axis.axhline(0, color="#555555", linewidth=0.8)
    energy_axis.set_ylabel("Solution energy (negative H-H contacts)")
    energy_axis.set_title("Solution energy (lower is better)")
    energy_axis.set_xticks(x, [name for name, _ in sequences])
    energy_axis.grid(axis="y", alpha=0.25)
    energy_axis.set_axisbelow(True)

    sa_time_means = [
        statistics.fmean(float(row["elapsed_seconds"]) for row in rows)
        for rows in grouped_sa
    ]
    sa_time_sds = [
        statistics.stdev(float(row["elapsed_seconds"]) for row in rows)
        if len(rows) > 1
        else 0.0
        for rows in grouped_sa
    ]
    agent_time_means = [float(result["time_mean"]) for result in agent_results]
    agent_time_sds = [float(result["time_sd"]) for result in agent_results]
    time_axis.bar(
        [position - width / 2 for position in x],
        [value - 0.001 for value in agent_time_means],
        width,
        bottom=0.001,
        yerr=agent_time_sds,
        capsize=5,
        color=agent_color,
        label=f"{model_name} (mean ± SD, 20 inferences)",
    )
    time_axis.bar(
        [position + width / 2 for position in x],
        [value - 0.001 for value in sa_time_means],
        width,
        bottom=0.001,
        yerr=sa_time_sds,
        capsize=5,
        color=sa_color,
        label="Simulated annealing (mean ± SD, 5 runs)",
    )
    for position, value in zip(x, agent_time_means):
        time_axis.annotate(
            f"{value:.3f}s",
            (position - width / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    for position, value in zip(x, sa_time_means):
        time_axis.annotate(
            f"{value:.3f}s",
            (position + width / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    time_axis.set_ylabel("Wall time (seconds)")
    time_axis.set_title("Solve / inference time (log scale; model load excluded)")
    time_axis.set_yscale("log")
    time_axis.set_ylim(bottom=0.001)
    time_axis.set_xticks(x, [name for name, _ in sequences])
    time_axis.grid(axis="y", alpha=0.25)
    time_axis.set_axisbelow(True)
    figure.legend(
        handles=[
            Line2D([0], [0], color=agent_color, lw=8, label=model_name),
            Line2D(
                [0],
                [0],
                color=sa_color,
                lw=8,
                label="Simulated annealing (100k, mean ± SD)",
            ),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        f"{model_name} best checkpoint vs 100k-iteration simulated annealing"
    )
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = load_latest_summary_rows()
    plot_training_curves(OUTPUT_DIR / "training_reward_eval_curves.png")
    plot_final_comparison(
        summary_rows, OUTPUT_DIR / "final_test_comparison.png"
    )

    best_row = max(summary_rows, key=lambda row: float(row["mean_contacts"]))
    network = best_row["network"]
    algorithm = best_row["algorithm"]
    model_name = f"{network.upper()}-{algorithm.upper()}"
    run_dir = RUN_DIR / f"{network}-{algorithm}" / f"seed{best_row['seed']}"
    model_class = DQN if algorithm == "dqn" else PPO
    model = model_class.load(str(run_dir / "best_model.zip"), device="cpu")
    test_rows = read_csv(run_dir / "test_folds.csv")
    sequences = benchmark_20mer_rows(
        test_rows, ROOT / "benchmark.txt"
    )
    agent_results = [
        infer_solution(model, network, sequence) for _, sequence in sequences
    ]
    plot_solution(
        sequences[0][0],
        model_name,
        sequences[0][1],
        agent_results[0],
        OUTPUT_DIR / "best_agent_20mer_solution.png",
    )
    plot_agent_vs_annealing(
        model_name,
        sequences,
        agent_results,
        read_csv(SA_RESULTS),
        OUTPUT_DIR / "best_agent_vs_100k_sa.png",
    )
    print(f"Best held-out pair: {model_name}")
    for (name, sequence), result in zip(sequences, agent_results):
        print(
            f"{name} ({sequence}): energy={result['energy']}, "
            f"contacts={result['contacts']}, "
            f"inference={result['time_mean']:.6f}s "
            f"(SD {result['time_sd']:.6f}s)"
        )
    print(f"Charts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
