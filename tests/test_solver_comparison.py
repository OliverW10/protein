import csv

import pytest

from compare_solvers import compare
from mcts import load_benchmark, search as mcts_search


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
