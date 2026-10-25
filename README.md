
# Reinforcement Learning for Protein Folding

## Setup

On a mac or linux computer setup with
```
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Simulated annealing

Run the root-level HP lattice solver with a sequence of H and P residues:

```sh
python simulated_annealing.py HPHPPHHPHH --iterations 100000 --seed 7
```

It prints the best fold found, its non-bonded H-H contact count, and energy
(negative contact count). Run `python simulated_annealing.py --help` to see
temperature and iteration options.

MCTS uses the same `--iterations` and `--seed` options and prints the same
fold metrics:

```sh
python mcts.py HPHPPHHPHH --iterations 100000 --seed 7
```

Compare both solvers on the shared benchmark sequence file with matched
iteration budgets and seeds:

```sh
python compare_solvers.py --iterations 100000 --seeds 7 14 21
```

The comparison prints mean contacts, completion rate, reference-energy hit
rate, and mean runtime, and saves per-sequence results to
`logs/solver_comparison.csv`. It also writes sampled best-so-far histories to
`logs/solver_comparison_progress.csv` and creates time-to-best and smoothed
energy plots beside the results CSV. `--progress-interval` controls sampling
frequency. Equal iteration budgets do not imply equal
computational work: an MCTS iteration includes a rollout, while an annealing
iteration proposes one local fold change.

On windows/powershell use:
```
python -m venv .env
.\.env\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Repeated simulated-annealing plots

Run five reproducible 150,000-iteration simulated-annealing runs per benchmark sequence and
generate an iteration-to-best bar chart with standard-deviation error bars,
plus unsmoothed best-so-far energy curves:

```sh
PYTHONPATH=. .venv/bin/python plot_simulated_annealing.py
```

The default output is `solver_results/150k_simulated_annealing/`. Override
the run count, iteration budget, seed range, or output directory with
`--runs`, `--iterations`, `--first-seed`, and `--output-dir`.