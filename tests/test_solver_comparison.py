import csv
from unittest.mock import patch

import pytest

from compare_solvers import compare
from mcts import load_benchmark, search as mcts_search
import plot_simulated_annealing
from plot_simulated_annealing import plot_results, run_benchmark


def test_mcts_accepts_lowercase_sequences_like_simulated_annealing():
    uppercase = mcts_search("HHHHHH", iterations=100, seed=5, show_progress=False)
    lowercase = mcts_search("hhhhhh", iterations=100, seed=5, show_progress=False)

    assert uppercase == lowercase


def test_benchmark_parser_reads_sequence_and_reference_energy(tmp_path):
    benchmark_path = tmp_path / "benchmark.txt"
    benchmark_path.write_text("4 hhhh -1\n", encoding="utf-8")

    assert load_benchmark(benchmark_path) == [
        ("sequence-01", "HHHH", -1)
    ]


def test_benchmark_parser_reports_invalid_sequence_with_location(tmp_path):
    benchmark_path = tmp_path / "benchmark.txt"
    benchmark_path.write_text("4 HHXP -1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"benchmark.txt:1: sequence"):
        load_benchmark(benchmark_path)


def test_both_solvers_write_matched_comparison_rows(tmp_path):
    benchmark_path = tmp_path / "benchmark.txt"
    benchmark_path.write_text("6 hhhhhh -2\n", encoding="utf-8")
    output_path = tmp_path / "results.csv"

    rows = compare(benchmark_path, output_path, iterations=200, seeds=[9], exploration=1.0)

    assert [row["algorithm"] for row in rows] == ["mcts", "simulated_annealing"]
    assert all(row["iterations"] == 200 for row in rows)
    assert all(row["seed"] == 9 for row in rows)
    assert all(row["complete"] for row in rows)
    with output_path.open(newline="", encoding="utf-8") as output_file:
        written_rows = list(csv.DictReader(output_file))
    assert len(written_rows) == 2
    assert written_rows[0]["sequence"] == written_rows[1]["sequence"] == "HHHHHH"
    assert all(float(row["time_to_final_best_seconds"]) >= 0 for row in rows)
    progress_path = tmp_path / "results_progress.csv"
    assert progress_path.exists()
    assert (tmp_path / "results_time_to_best.png").exists()
    assert (tmp_path / "results_smoothed_energy.png").exists()
    with progress_path.open(newline="", encoding="utf-8") as progress_file:
        progress_rows = list(csv.DictReader(progress_file))
    assert {row["algorithm"] for row in progress_rows} == {
        "mcts",
        "simulated_annealing",
    }


def test_repeated_annealing_run_saves_iteration_data_and_charts(tmp_path):
    benchmark_path = tmp_path / "benchmark.txt"
    benchmark_path.write_text("6 hhhhhh -2\n", encoding="utf-8")

    results, progress = run_benchmark(
        benchmark_path,
        tmp_path / "plots",
        iterations=20,
        runs=2,
        first_seed=11,
        progress_interval=5,
    )

    assert len(results) == 2
    assert {row["seed"] for row in results} == {11, 12}
    assert all(row["time_to_final_best_iterations"] <= 20 for row in results)
    assert {row["run"] for row in progress} == {1, 2}
    assert (tmp_path / "plots/simulated_annealing_results.csv").exists()
    assert (tmp_path / "plots/simulated_annealing_progress.csv").exists()
    assert (
        tmp_path / "plots/simulated_annealing_time_to_best_iterations.png"
    ).exists()
    assert (
        tmp_path / "plots/simulated_annealing_best_so_far_energy.png"
    ).exists()


def test_annealing_plots_show_names_without_reference_energy_lines(tmp_path):
    results = [
        {
            "name": "sequence-01",
            "sequence": "HHHH",
            "iterations": 10,
            "run": 1,
            "time_to_final_best_iterations": 5,
            "reference_energy": -2,
        }
    ]
    progress = [
        {
            "name": "sequence-01",
            "run": 1,
            "step": 0,
            "best_energy": 0,
        },
        {
            "name": "sequence-01",
            "run": 1,
            "step": 10,
            "best_energy": -1,
        },
        {
            "name": "sequence-01",
            "run": 2,
            "step": 0,
            "best_energy": 0,
        },
        {
            "name": "sequence-01",
            "run": 2,
            "step": 10,
            "best_energy": -2,
        },
    ]
    original_subplots = plot_simulated_annealing.plt.subplots
    figures = []

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        figures.append(figure)
        return figure, axes

    with patch("plot_simulated_annealing.plt.subplots", side_effect=capture_subplots):
        plot_results(results, progress, tmp_path, progress_interval=10)

    assert [label.get_text() for label in figures[0].axes[0].get_yticklabels()] == [
        "sequence-01"
    ]
    energy_axis = figures[1].axes[0]
    assert len(energy_axis.lines) == 1
    assert energy_axis.lines[0].get_linestyle() == "-"
    assert energy_axis.get_legend().get_title().get_text() == "Benchmark sequence"
    assert list(energy_axis.lines[0].get_ydata()) == [0, -2]
