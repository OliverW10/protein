# 1,000,000-iteration simulated-annealing benchmark

This directory contains the benchmark input, per-run result rows, best-so-far
progress traces, and charts. It contains five independent runs per sequence,
with seeds 7 through 11 and 1,000,000 iterations per run.

Reproduce from the repository root:

```sh
PYTHONPATH=. .venv/bin/python plot_simulated_annealing.py \
  --benchmark solver_results/1m_simulated_annealing/benchmark.txt \
  --output-dir solver_results/1m_simulated_annealing \
  --iterations 1000000 \
  --runs 5 \
  --first-seed 7 \
  --progress-interval 1000
```

The benchmark and chart generator is [`plot_simulated_annealing.py`](../../plot_simulated_annealing.py),
using the solver in [`simulated_annealing.py`](../../simulated_annealing.py).
`simulated_annealing_results.csv` contains one row per sequence/run;
`simulated_annealing_progress.csv` contains the best-so-far updates and
interval samples used to build the energy chart.

The iteration bar chart labels each row with its benchmark sequence name and
shows the mean number of iterations to first reach that run's final best
energy, with sample-standard-deviation error bars across runs. The energy
chart shows one unsmoothed best-so-far energy curve per sequence, taking the
minimum energy across the five runs at each sampled iteration. Curves use a
viridis color gradient. The best-known attainment chart reports mean contacts
per run divided by the benchmark's known contact count, shown as percentages
with sample-standard-deviation error bars; values above 100% indicate that a
run exceeded the reference result.
