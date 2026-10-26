# Combined architecture and algorithm benchmark

The experiment runner compares six combinations on random 20-residue HP
sequences:

- FC-DQN
- LSTM-DQN
- CNN-DQN
- FC-PPO
- LSTM-PPO
- CNN-PPO

The lattice package now returns only its flattened grid and does not apply
distance shaping. `Oliver.envs` supplies the distance, trap-correction, and
fold-metric wrappers; `Muhammad.dqn.envs.SequenceObservation` adds the sequence
and valid-move features used by the LSTM extractor. PPO uses the same matching
`CustomANN`, `CustomLSTM`, or `CustomCNN` feature extractor as DQN. The LSTM
extractor reads the full current sequence/action representation at each
decision, so it does not require a recurrent PPO policy or `sb3-contrib`. For
faster experiments, Combined uses four DQN gradient steps per episode (one in
smoke mode); this is a training configuration change that should be evaluated
for its effect on learning quality.

The CNN uses Willie's five-channel lattice representation (H/P occupancy,
residue index, next-residue type, and fold progress) with a custom convolutional
feature extractor. Select it with `cnn` in `NETWORKS`; CNN environments omit
the flattened FC lookahead because the next-residue feature is encoded as an
image channel. Its smaller extractor uses two convolutional layers with 16 and
32 channels, followed by a 128-dimensional feature layer.

To run a short end-to-end check, set `SMOKE = True` near the top of
`Combined/train.py`, then run:

```sh
python -m Combined.train
```

By default, smoke mode trains one seed for 256 environment timesteps per
combination, evaluates two validation sequences every episode, and tests on
the two fixed benchmark sequences plus two random test sequences.

For the full comparison, leave `SMOKE = False` and run:

```sh
python -m Combined.train
```

This trains each combination toward 100,000 environment steps for each of
seeds 7, 14, and 21. SB3 checks the target between rollouts, so the final
count may slightly exceed the requested target. Validation still runs every
200 completed training episodes; this is an evaluation interval, not the
training budget. The best validation checkpoint is tested on 100 seeded
random sequences and the two fixed benchmark sequences. To select a subset or
alter run sizes, edit `NETWORKS`, `ALGORITHMS`, `SEEDS`, `TOTAL_TIMESTEPS`,
`NUM_VALIDATION_SEQUENCES`, `EVALUATION_INTERVAL`, `NUM_TEST_SEQUENCES`,
`DEVICE`, `THREADS`, or `OUTPUT_DIR` in `Combined/train.py`.

FC environments include a 20-residue lookahead by default. It is implemented
by `Oliver.envs.NextResiduesObservation` and then flattened for the FC policy.
Pass `include_next_residues=False` to `Oliver.envs.make_env` to use only the
grid. The LSTM environment omits this adapter because its sequence observation
already contains the full residue sequence.

Outputs are written under `logs/combined`: each run stores checkpoints,
TensorBoard logs, validation history, the final and best models, per-sequence
test folds, and design settings. `benchmark_summary.csv` collects the
per-run test metrics for direct comparison.

Generate charts for the latest completed six-pair run, including TensorBoard
training reward with validation overlays, held-out results, the best agent's
20-residue fold, and a comparison with the 100k simulated-annealing runs:

```sh
python -m Combined.generate_latest_charts
```

The PNGs are written to `Combined/results/latest_run/`.

Generate a Markdown report and validation/test charts from the saved smoke and
partial-run outputs:

```sh
python -m Combined.summarize_results
```

The report and PNG charts are written to `Combined/results/`. Pass
`--smoke-dir`, `--partial-dir`, or `--output-dir` to use different input or
output locations.
