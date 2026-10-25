import random
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd

from .callbacks import greedy_episode
from .envs import make_env, moves_to_coords, random_sequence_fn
from .train import num_test, seq_length, sequences, test_seed


# The 20mer-A/B benchmarks plus the random test proteins (never used in training or validation)
def get_test_sequences():

    test_seqs = OrderedDict()
    for name, info in sequences.items():
        test_seqs[name] = info["seq"]

    next_test = random_sequence_fn(seq_length, test_seed)
    for i in range(num_test):
        test_seqs[f"test-{i:03d}"] = next_test()

    return test_seqs


# Fold every test protein greedily with a trained model (one row per protein)
def evaluate_model(model, net, test_seqs):

    rows = []
    for name, seq in test_seqs.items():
        # The LSTM needs the sequence view and the fully connected DQN the grid
        env = make_env(seq, sequence_view=(net == "lstm"))
        result = greedy_episode(model, env)
        rows.append({"name": name, "contacts": result["contacts"], "complete": result["complete"]})

    return pd.DataFrame(rows)


# Random moves on the same proteins keeping the best complete fold out of tries attempts
def random_baseline(test_seqs, tries=100, seed=0):

    rng = random.Random(seed)
    rows = []
    for name, seq in test_seqs.items():
        env = make_env(seq)
        best_contacts = 0
        complete = False
        for attempt in range(tries):
            env.reset()
            while True:
                _, _, terminated, truncated, info = env.step(rng.randint(0, 3))
                if terminated or truncated:
                    break

            # Only complete folds count
            if info["complete"]:
                complete = True
                best_contacts = max(best_contacts, info["contacts"])

        rows.append({"name": name, "contacts": best_contacts, "complete": complete})

    return pd.DataFrame(rows)


# One row per model with its average on the random test proteins and its energy on each benchmark
def results_table(evaluations):

    rows = []
    for label, results in evaluations.items():
        # Incomplete folds count as 0 contacts
        scores = results["contacts"].where(results["complete"], 0)
        is_test = results["name"].str.startswith("test")
        row = {
            "Model": label,
            "Test Contacts": scores[is_test].mean(),
            "Test Complete": results["complete"][is_test].mean(),
        }

        # Benchmarks are given as energies so they line up with the best known ones
        for _, protein in results[~is_test].iterrows():
            row[protein["name"]] = -protein["contacts"] if protein["complete"] else None
        rows.append(row)

    best_known = {"Model": "Best Known"}
    for name in rows[0]:
        if name in sequences:
            best_known[name] = sequences[name]["best_energy"]
    rows.append(best_known)

    return pd.DataFrame(rows)


# Print the env's own terminal picture cut down to the rows and columns around the chain
def render_chain(env):

    text = env.unwrapped.render(mode="ansi").getvalue()
    lines = text.split("\n")

    # Every cell ends with the colour reset code so splitting on it gives one piece per cell
    grid = [line.split("\x1b[0m")[:-1] for line in lines[1:] if line != ""]
    rows = [i for i, row in enumerate(grid) if any(cell[-1] in "HP" for cell in row)]
    cols = [j for row in grid for j, cell in enumerate(row) if cell[-1] in "HP"]
    top = max(min(rows) - 1, 0)
    bottom = min(max(rows) + 1, len(grid) - 1)
    left = max(min(cols) - 1, 0)
    right = min(max(cols) + 1, len(grid[0]) - 1)

    # The first line is the last move like (Left)
    print(lines[0])
    for row in grid[top:bottom + 1]:
        print("".join(cell + "\x1b[0m" for cell in row[left:right + 1]))


# Draw a fold plot
def draw_fold(fold, seq, title=""):

    coords = moves_to_coords(fold)
    xs = [x for x, y in coords]
    ys = [y for x, y in coords]

    plt.figure(figsize=(5, 5))
    plt.plot(xs, ys, color="grey", linewidth=3, zorder=1)

    index_at = {}
    for i, position in enumerate(coords):
        index_at[position] = i
    for i, (x, y) in enumerate(coords):
        if seq[i] != "H":
            continue
        for neighbour in [(x + 1, y), (x, y + 1)]:
            j = index_at.get(neighbour)
            if j is not None and seq[j] == "H" and abs(i - j) > 1:
                plt.plot([x, neighbour[0]], [y, neighbour[1]], color="red", linestyle=":", linewidth=2, zorder=1)

    for i, (x, y) in enumerate(coords):
        colour = "black" if seq[i] == "H" else "white"
        plt.scatter(x, y, s=300, c=colour, edgecolors="black", zorder=2)
        plt.text(x, y, seq[i], color="white" if seq[i] == "H" else "black", ha="center", va="center",
                 fontsize=9, fontweight="bold", zorder=4)

    plt.scatter(xs[0], ys[0], s=700, facecolors="none", edgecolors="blue", zorder=3)

    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.title(title)
    plt.show()


# Replay a finished fold then show the env's terminal picture first and the drawn picture second
def show_fold(fold, seq, title=""):

    actions = {"L": 0, "D": 1, "U": 2, "R": 3}
    env = make_env(seq)
    env.reset()
    for move in fold:
        env.step(actions[move])

    print(title)
    render_chain(env)
    draw_fold(fold, seq, title)


# Watch a trained agent fold one protein with the env's terminal picture after every move
def watch_agent(model, net, seq):

    env = make_env(seq, sequence_view=(net == "lstm"))
    obs, _ = env.reset()
    render_chain(env)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        render_chain(env)
        if terminated or truncated:
            break

    print("Complete:", info["complete"], "| Contacts:", info["contacts"], "| Collisions:", info["collisions"],
          "| Moves:", "".join(info["actions"]))
