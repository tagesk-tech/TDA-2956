"""Export the tested analysis arrays as the six report figures."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ripser import ripser

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(ROOT))
from data import ACTIVITIES, load_split
from tda import GRID, delay_cloud, mean_autocorrelation, normalize_cloud


results = json.loads((ROOT / "output/results.json").read_text())
delay = np.load(ROOT / "output/cache/delay_m3_lag15_f47.npz")
train, labels, subjects = load_split("train")
colors = plt.cm.tab10(np.linspace(0, 1, 6))


def save(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, bbox_inches="tight")
    plt.close()


# Lag selection.
autocorrelation = mean_autocorrelation(train["body_acc"], 40)
plt.figure(figsize=(6.8, 3.2))
plt.plot(range(41), autocorrelation, marker="o", markersize=2.5)
plt.axhline(0, color="black", linewidth=0.7)
plt.axvline(15, color="tab:red", linestyle="--", label=r"$\ell=15$")
plt.xlabel("Lag (samples)")
plt.ylabel("Mean normalized autocorrelation")
plt.legend()
save("lag.pdf")


# One delay cloud and its persistence diagrams.
index = np.flatnonzero((subjects == 1) & (labels == 1))[0]
cloud, _ = normalize_cloud(delay_cloud(train["body_acc"][index], 3, 15))
diagrams = ripser(cloud, maxdim=1, coeff=47)["dgms"]
fig = plt.figure(figsize=(10, 3.4))
axis = fig.add_subplot(1, 3, 1, projection="3d")
axis.plot(cloud[:, 0], cloud[:, 1], cloud[:, 2], marker="o", markersize=1.5)
axis.set(title="Delay cloud (first 3 coordinates)", xlabel="$z_1$", ylabel="$z_2$")
for dimension in (0, 1):
    axis = fig.add_subplot(1, 3, dimension + 2)
    finite = diagrams[dimension][np.isfinite(diagrams[dimension][:, 1])]
    limit = max(1.0, float(finite.max()) * 1.05)
    axis.scatter(finite[:, 0], finite[:, 1], s=12)
    axis.plot((0, limit), (0, limit), color="black", linewidth=0.7)
    axis.set(title=rf"$H_{dimension}$ persistence", xlabel="Birth", ylabel="Death")
save("construction.pdf")


# Mean stable ranks for each activity and sensor/homology block.
ranks = delay["train_ranks"]
fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True)
titles = ("Acceleration $H_0$", "Acceleration $H_1$",
          "Gyroscope $H_0$", "Gyroscope $H_1$")
for block, (axis, title) in enumerate(zip(axes.flat, titles)):
    start = block * len(GRID)
    for (activity, activity_name), color in zip(ACTIVITIES.items(), colors):
        mean = ranks[labels == activity, start:start + len(GRID)].mean(axis=0)
        axis.plot(GRID, mean, label=activity_name, color=color)
    axis.set(title=title, xlabel="Lifetime threshold", ylabel="Mean stable rank")
axes[0, 0].legend(fontsize=6.5)
save("stable_ranks.pdf")


# Fixed-protocol model comparison.
names = list(results["models"])
development = [results["models"][name]["development"]["balanced_accuracy"] for name in names]
official = [results["models"][name]["official_test"]["balanced_accuracy"] for name in names]
x = np.arange(len(names))
plt.figure(figsize=(8.5, 3.8))
plt.bar(x - 0.18, development, 0.36, label="Grouped development")
plt.bar(x + 0.18, official, 0.36, label="Official test")
plt.axhline(1 / 6, color="black", linestyle=":", label="Chance")
plt.xticks(x, names, rotation=23, ha="right")
plt.ylabel("Balanced accuracy")
plt.ylim(0, 1)
plt.legend(fontsize=8)
save("classification.pdf")


# Cross-subject activity geometry.
matrix = np.asarray(results["activity_distance_matrix"])
plt.figure(figsize=(5.8, 4.8))
plt.imshow(matrix, cmap="magma")
plt.colorbar(label="Mean stable-rank distance")
names = list(ACTIVITIES.values())
plt.xticks(range(6), names, rotation=32, ha="right")
plt.yticks(range(6), names)
save("activity_geometry.pdf")


# Interpretable circular features.
dynamics = delay["train_dynamics"]
columns = (6, 7, 14, 15)
titles = ("Acceleration frequency", "Acceleration linearity",
          "Gyroscope frequency", "Gyroscope linearity")
fig, axes = plt.subplots(2, 2, figsize=(9, 5.8))
for axis, column, title in zip(axes.flat, columns, titles):
    values = [dynamics[labels == activity, column].mean() for activity in ACTIVITIES]
    axis.bar(range(6), values, color=colors)
    axis.set_title(title)
    axis.set_xticks(range(6), names, rotation=25, ha="right")
save("circular_dynamics.pdf")

print(f"Wrote report figures to {FIGURES}")
