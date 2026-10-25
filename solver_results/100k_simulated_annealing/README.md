# Repeated simulated-annealing benchmark

This directory contains the benchmark input, per-run result rows, best-so-far
progress traces, and the two regenerated charts. It contains five independent
runs per sequence, with seeds 7 through 11 and 100,000 iterations per run.

Reproduce from the repository root:

```sh
PYTHONPATH=. .venv/bin/python plot_simulated_annealing.py \
  --benchmark solver_results/100k_simulated_annealing/benchmark.txt \
  --output-dir solver_results/100k_simulated_annealing \
  --iterations 100000 \
  --runs 5 \
  --first-seed 7 \
  --progress-interval 1000
```

The run and plotting implementation is [`plot_simulated_annealing.py`](../../plot_simulated_annealing.py),
using the solver in [`simulated_annealing.py`](../../simulated_annealing.py).
`simulated_annealing_results.csv` contains one row per sequence/run;
`simulated_annealing_progress.csv` contains the unsmoothed best-so-far updates
and interval samples used to build the energy plot.

The iteration bar chart uses the full HP sequence as its row label and shows
the mean number of iterations to first reach that run's final best energy,
with sample-standard-deviation error bars across runs. The energy chart shows
each sequence's mean best-so-far energy over runs without smoothing. The
sequence traces use distinct colors from a viridis gradient.
