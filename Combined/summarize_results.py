"""Create a concise report and Matplotlib charts from saved benchmark outputs."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMOKE_DIR = ROOT / "logs" / "combined-smoke"
DEFAULT_PARTIAL_DIR = ROOT / "logs" / "combined"
DEFAULT_OUTPUT_DIR = ROOT / "Combined" / "results"

MODEL_ORDER = [
    ("fc", "dqn", "FC-DQN"),
    ("fc", "ppo", "FC-PPO"),
    ("lstm", "dqn", "LSTM-DQN"),
    ("lstm", "ppo", "LSTM-PPO"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def load_histories(smoke_dir: Path, partial_dir: Path):
    histories = {}
    for network, algorithm, label in MODEL_ORDER:
        path = smoke_dir / f"{network}-{algorithm}" / "seed7" / "eval_history.csv"
        histories[label] = read_csv(path) if path.exists() else []

    partial_path = partial_dir / "fc-dqn" / "seed7" / "eval_history.csv"
    partial = read_csv(partial_path) if partial_path.exists() else []
    return histories, partial


def plot_training(histories, partial, output_path: Path) -> None:
    figure, (partial_axis, smoke_axis) = plt.subplots(
        1, 2, figsize=(12, 4.8), constrained_layout=True
    )

    if partial:
        episodes = [int(row["episodes"]) for row in partial]
        scores = [float(row["score"]) for row in partial]
        partial_axis.plot(episodes, scores, marker="o", color="#3366a3")
        partial_axis.set_title("FC-DQN partial full-run validation")
        partial_axis.set_xlabel("Training episodes")
        partial_axis.set_ylabel("Mean validation contacts")
        partial_axis.set_xlim(left=0)
        partial_axis.set_ylim(bottom=0)
        partial_axis.grid(alpha=0.25)
    else:
        partial_axis.text(
            0.5, 0.5, "No partial full-run validation data found",
            ha="center", va="center", transform=partial_axis.transAxes,
        )
        partial_axis.set_axis_off()

    for label, rows in histories.items():
        if rows:
            episodes = [int(row["episodes"]) for row in rows]
            scores = [float(row["score"]) for row in rows]
            smoke_axis.plot(episodes, scores, marker="o", label=label)
    smoke_axis.set_title("Smoke-run validation (3 episodes each)")
    smoke_axis.set_xlabel("Training episodes")
    smoke_axis.set_ylabel("Mean validation contacts")
    smoke_axis.set_xlim(left=0)
    smoke_axis.set_ylim(bottom=0)
    smoke_axis.set_xticks([1, 2, 3])
    smoke_axis.grid(alpha=0.25)
    smoke_axis.legend(frameon=False, fontsize="small")

    figure.suptitle("Training validation curves — exploratory runs only")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_test_results(rows: list[dict[str, str]], output_path: Path) -> None:
    labels = [
        f"{row['network'].upper()}-{row['algorithm'].upper()}" for row in rows
    ]
    mean_contacts = [float(row["mean_contacts"]) for row in rows]
    completion = [float(row["completion_rate"]) * 100 for row in rows]
    colors = ["#4477aa", "#66ccee", "#228833", "#ccbb44"]

    figure, (contacts_axis, completion_axis) = plt.subplots(
        1, 2, figsize=(10, 4.8), constrained_layout=True
    )
    for axis, values, title, ylabel in (
        (contacts_axis, mean_contacts, "Random test folds", "Mean H-H contacts"),
        (completion_axis, completion, "Episode completion", "Completed episodes (%)"),
    ):
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.2f" if axis is contacts_axis else "%.0f%%")
        axis.set_ylim(0, max(values, default=0) + (0.1 if axis is contacts_axis else 10))

    figure.suptitle("Smoke benchmark test metrics (2 random sequences per model)")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def create_summary(
    smoke_dir: Path,
    partial_dir: Path,
    output_dir: Path,
) -> Path:
    summary_path = smoke_dir / "benchmark_summary.csv"
    rows = read_csv(summary_path)
    if not rows:
        raise ValueError(f"No benchmark summary rows found in {summary_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    training_chart = output_dir / "training_validation.png"
    test_chart = output_dir / "smoke_test_results.png"
    histories, partial = load_histories(smoke_dir, partial_dir)
    plot_training(histories, partial, training_chart)
    plot_test_results(rows, test_chart)

    lines = [
        "# Protein-folding experiment results",
        "",
        "## Scope and status",
        "",
        "- All four architecture/algorithm combinations completed the smoke run:",
        "  FC-DQN, FC-PPO, LSTM-DQN, and LSTM-PPO.",
        "- Each smoke run used seed 7, trained for 3 episodes, evaluated 2",
        "  validation sequences after each episode, and tested 2 random sequences",
        "  in addition to the 2 fixed benchmark sequences.",
        "- The full 100,000-episode, three-seed matrix is **not complete**.",
        "  Only FC-DQN seed 7 has a partial saved full-run validation trace.",
        "",
        "## Smoke test results",
        "",
        "| Model | Timesteps | Mean contacts (2 random tests) | Completion rate |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['network'].upper()}-{row['algorithm'].upper()} "
            f"| {int(row['timesteps']):,} "
            f"| {float(row['mean_contacts']):.2f} "
            f"| {float(row['completion_rate']):.0%} |"
        )

    fixed_test_rows = {}
    for network, algorithm, label in MODEL_ORDER:
        path = smoke_dir / f"{network}-{algorithm}" / "seed7" / "test_folds.csv"
        if path.exists():
            fixed_test_rows[label] = read_csv(path)

    lines.extend(
        [
            "",
            "The smoke models completed both random test episodes (100% completion)",
            "but achieved 0.00 mean H-H contacts across the two random test",
            "sequences. The short run and tiny test sample are only a pipeline",
            "check, not evidence of learned folding performance.",
            "",
            "### Fixed benchmark sequences",
            "",
            "| Model | 20mer-A energy | 20mer-B energy |",
            "|---|---:|---:|",
        ]
    )
    for _, _, label in MODEL_ORDER:
        energy_by_name = {
            row["name"]: row["energy"] for row in fixed_test_rows.get(label, [])
        }
        energy_a = energy_by_name.get("20mer-A") or "incomplete"
        energy_b = energy_by_name.get("20mer-B") or "incomplete"
        lines.append(f"| {label} | {energy_a} | {energy_b} |")

    if partial:
        best = max(float(row["score"]) for row in partial)
        last = partial[-1]
        lines.extend(
            [
                "",
                "## Partial full-run training trace",
                "",
                f"FC-DQN seed 7 has validation entries through "
                f"{int(last['episodes']):,} episodes. The best recorded mean "
                f"validation score was {best:.2f} contacts; at the last "
                f"recorded point it was {float(last['score']):.2f}. This is "
                "an incomplete run and must not be compared as a trained model.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Partial full-run training trace",
                "",
                "No full-run validation trace was available.",
            ]
        )

    lines.extend(
        [
            "",
            "## Charts",
            "",
            f"![Training validation curves](./{training_chart.name})",
            "",
            f"![Smoke test results](./{test_chart.name})",
            "",
            "The validation plots show the incomplete FC-DQN trace separately",
            "from the three-episode smoke curves. Test-metric bars summarize",
            "only the two random test sequences per smoke model.",
            "",
        ]
    )
    report_path = output_dir / "results_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown result summary and Matplotlib charts."
    )
    parser.add_argument("--smoke-dir", type=Path, default=DEFAULT_SMOKE_DIR)
    parser.add_argument("--partial-dir", type=Path, default=DEFAULT_PARTIAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = create_summary(args.smoke_dir, args.partial_dir, args.output_dir)
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
