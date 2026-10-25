# 100,000-iteration solver comparison

This directory preserves the input data, raw results, progress traces, and
plots from the benchmark comparison.

## Reproduce

From the repository root, run:

```sh
PYTHONPATH=. .venv/bin/python compare_solvers.py \
  --benchmark solver_results/100k/benchmark.txt \
  --iterations 100000 \
  --seeds 7 \
  --progress-interval 1000 \
  --output solver_results/100k/solver_comparison_100k.csv
```

The plotting and comparison code is in [`compare_solvers.py`](../../compare_solvers.py);
the algorithms are in [`mcts.py`](../../mcts.py) and
[`simulated_annealing.py`](../../simulated_annealing.py). The benchmark input
is copied here so the run can be reproduced independently of later edits to
the root [`benchmark.txt`](../../benchmark.txt).

## Files

- `solver_comparison_100k.csv`: one final result per solver and sequence.
- `solver_comparison_100k_progress.csv`: best-so-far energies sampled during
  each run.
- `solver_comparison_100k_time_to_best.png`: elapsed time to first reach each
  run's final best energy.
- `solver_comparison_100k_smoothed_energy.png`: mean best-so-far energy across
  sequences, smoothed with a centered moving average.
- `benchmark.txt`: the exact input sequences and reference energies.

Each of the 15 sequences was run once per algorithm with seed 7 and 100,000
iterations. MCTS averaged 12.333 contacts, 7.887 seconds per run, and 3.406
seconds to its final best energy. Simulated annealing averaged 14.733
contacts, 0.881 seconds per run, and 0.346 seconds to its final best energy.
Both algorithms completed all sequences; neither reached the benchmark
reference energies in this run.
