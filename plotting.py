"""Plots used during exploratory analysis of UCI HAR windows."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from data_check import ACTIVITY_NAMES


def plot_time_series(window, title: str) -> None:
    """Plot the three acceleration coordinates through one window."""
    plt.plot(window)
    plt.xlabel("Time sample")
    plt.ylabel("Body acceleration")
    plt.legend(["x", "y", "z"])
    plt.title(title)
    plt.show()


def plot_activity_trajectories(signals, labels, subjects, subject: int, output_path=None) -> None:
    """Plot one three-dimensional trajectory per activity for one subject."""
    windows = {}

    for activity in ACTIVITY_NAMES:
        indices = np.flatnonzero(
            (subjects == subject) & (labels == activity)
        )
        windows[activity] = signals[indices[0]]

    all_points = np.concatenate(list(windows.values()))
    lower_limits = all_points.min(axis=0)
    upper_limits = all_points.max(axis=0)
    figure = plt.figure(figsize=(13, 8))

    for activity in ACTIVITY_NAMES:
        window = windows[activity]

        axis = figure.add_subplot(2, 3, activity, projection="3d")
        axis.plot(window[:, 0], window[:, 1], window[:, 2])
        axis.set_title(ACTIVITY_NAMES[activity])
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.set_xlim(lower_limits[0], upper_limits[0])
        axis.set_ylim(lower_limits[1], upper_limits[1])
        axis.set_zlim(lower_limits[2], upper_limits[2])

    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def plot_mean_autocorrelation(autocorrelation, chosen_lag: int, output_path=None) -> None:
    """Plot the mean autocorrelation and its first zero crossing."""
    lags = np.arange(len(autocorrelation))
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(lags, autocorrelation)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(chosen_lag, color="tab:red", linestyle="--", label=f"lag = {chosen_lag}")
    axis.set_xlabel("Lag (samples)")
    axis.set_ylabel("Mean autocorrelation")
    axis.legend()
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def plot_activity_persistence(diagrams, output_path=None) -> None:
    """Plot H0 and H1 persistence diagrams with common axes."""
    finite = [
        point
        for activity_diagrams in diagrams.values()
        for diagram in activity_diagrams
        for point in diagram
        if np.isfinite(point[1])
    ]
    limit = 1.05 * np.max(finite)
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)

    for axis, activity in zip(axes.flat, ACTIVITY_NAMES):
        for dimension, color in ((0, "tab:blue"), (1, "tab:orange")):
            diagram = diagrams[activity][dimension]
            diagram = diagram[np.isfinite(diagram[:, 1])]
            axis.scatter(
                diagram[:, 0], diagram[:, 1], s=15, alpha=0.7,
                color=color, label=f"$H_{dimension}$",
            )
        axis.plot([0, limit], [0, limit], color="black", linewidth=0.8)
        axis.set_title(ACTIVITY_NAMES[activity])
        axis.set_aspect("equal")

    axes[0, 0].legend()
    figure.supxlabel("Birth")
    figure.supylabel("Death")
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def plot_stable_rank_means(features, labels, subjects, grid, output_path=None) -> None:
    """Plot subject-balanced activity means and uncertainty across subjects."""
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    titles = (
        "Acceleration $H_0$", "Acceleration $H_1$",
        "Gyroscope $H_0$", "Gyroscope $H_1$",
    )
    width = len(grid)

    for block, (axis, title) in enumerate(zip(axes.flat, titles)):
        values = features[:, block * width : (block + 1) * width]
        for activity, name in ACTIVITY_NAMES.items():
            subject_means = np.stack([
                values[(subjects == subject) & (labels == activity)].mean(axis=0)
                for subject in np.unique(subjects)
            ])
            mean = subject_means.mean(axis=0)
            standard_error = subject_means.std(axis=0, ddof=1) / np.sqrt(
                len(subject_means)
            )
            axis.plot(grid, mean, label=name)
            axis.fill_between(grid, mean - standard_error, mean + standard_error, alpha=0.12)
        axis.set_title(title)
        axis.set_ylabel("Stable rank")

    axes[-1, 0].set_xlabel("Lifetime threshold")
    axes[-1, 1].set_xlabel("Lifetime threshold")
    axes[0, 1].legend(fontsize=8)
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def plot_joint_stable_rank_means(features, labels, subjects, grid, output_path=None) -> None:
    """Plot subject-balanced H0/H1 means for the joint sensor cloud."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    width = len(grid)

    for block, (axis, title) in enumerate(zip(axes, ("Joint $H_0$", "Joint $H_1$"))):
        values = features[:, block * width : (block + 1) * width]
        for activity, name in ACTIVITY_NAMES.items():
            subject_means = np.stack([
                values[(subjects == subject) & (labels == activity)].mean(axis=0)
                for subject in np.unique(subjects)
            ])
            mean = subject_means.mean(axis=0)
            standard_error = subject_means.std(axis=0, ddof=1) / np.sqrt(
                len(subject_means)
            )
            axis.plot(grid, mean, label=name)
            axis.fill_between(grid, mean - standard_error, mean + standard_error, alpha=0.12)
        axis.set_title(title)
        axis.set_xlabel("Lifetime threshold")
        axis.set_ylabel("Stable rank")

    axes[1].legend(fontsize=8)
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def plot_joint_analysis(results, output_path=None) -> None:
    """Compare joint feature sets and acceleration-weight sensitivity."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    score_names = (
        "Separate sensor topology",
        "Joint sensor topology",
        "Separate + joint topology",
    )
    score_values = [results["scores"][name]["balanced_accuracy"] for name in score_names]
    axes[0].bar(score_names, score_values)
    axes[0].axhline(1 / 6, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Grouped-CV balanced accuracy")
    axes[0].set_ylim(0, 0.65)
    axes[0].tick_params(axis="x", rotation=20)

    weight_names = list(results["weight_sensitivity"])
    weights = [float(name.rsplit(" ", 1)[-1]) for name in weight_names]
    weight_scores = [
        results["weight_sensitivity"][name]["balanced_accuracy"]
        for name in weight_names
    ]
    axes[1].plot(weights, weight_scores, marker="o")
    axes[1].set_xlabel("Acceleration weight")
    axes[1].set_ylabel("Grouped-CV balanced accuracy")
    axes[1].set_ylim(0, 0.65)

    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def plot_synchronization_control(results, output_path=None) -> None:
    """Compare correct pairing with five broken-pairing realizations."""
    control = results["desynchronization_control"]
    seed_scores = control["seed_scores"]
    synchronized = np.asarray((
        results["scores"]["Joint sensor topology"]["balanced_accuracy"],
        results["scores"]["Separate + joint topology"]["balanced_accuracy"],
    ))
    desynchronized = np.asarray([
        (
            scores["joint"]["balanced_accuracy"],
            scores["separate_plus_joint"]["balanced_accuracy"],
        )
        for scores in seed_scores.values()
    ])

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    positions = np.arange(2)
    axis.scatter(
        positions - 0.08,
        synchronized,
        marker="s",
        s=75,
        label="Correct pairing",
        zorder=3,
    )
    for row, seed in enumerate(seed_scores):
        axis.scatter(
            positions + 0.08,
            desynchronized[row],
            color="tab:orange",
            alpha=0.75,
            label="Broken pairing (five seeds)" if row == 0 else None,
            zorder=2,
        )
    axis.plot(
        positions + 0.08,
        desynchronized.mean(axis=0),
        linestyle="none",
        marker="_",
        markersize=20,
        markeredgewidth=2,
        color="black",
        label="Broken-pairing mean",
        zorder=4,
    )
    axis.set_xticks(positions, ("Joint only", "Separate + joint"))
    axis.set_ylabel("Grouped-CV balanced accuracy")
    axis.set_ylim(0.38, 0.60)
    axis.legend()
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def plot_coupled_max_analysis(results, output_path=None) -> None:
    """Summarize the max-product filtration and its pairing control."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    score_names = (
        "Separate",
        "Euclidean\njoint",
        "Max\ncoupled",
        "Separate +\nmax coupled",
    )
    score_keys = (
        "Separate sensor topology",
        "Euclidean joint topology",
        "Max-coupled topology",
        "Separate + max-coupled topology",
    )
    axes[0].bar(
        score_names,
        [results["scores"][name]["balanced_accuracy"] for name in score_keys],
    )
    axes[0].axhline(1 / 6, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Grouped-CV balanced accuracy")
    axes[0].set_ylim(0, 0.62)
    axes[0].set_title("Representations")

    weight_names = list(results["weight_sensitivity"])
    weights = [float(name.rsplit(" ", 1)[-1]) for name in weight_names]
    weight_scores = [
        results["weight_sensitivity"][name]["balanced_accuracy"]
        for name in weight_names
    ]
    axes[1].plot(weights, weight_scores, marker="o")
    axes[1].set_xlabel("Acceleration weight")
    axes[1].set_ylim(0, 0.62)
    axes[1].set_title("Metric slice")

    synchronized = np.asarray((
        results["scores"]["Max-coupled topology"]["balanced_accuracy"],
        results["scores"]["Separate + max-coupled topology"]["balanced_accuracy"],
    ))
    broken = np.asarray([
        (
            scores["coupled"]["balanced_accuracy"],
            scores["separate_plus_coupled"]["balanced_accuracy"],
        )
        for scores in results["desynchronization_seed_scores"].values()
    ])
    positions = np.arange(2)
    axes[2].scatter(
        positions - 0.08,
        synchronized,
        marker="s",
        s=75,
        label="Correct pairing",
        zorder=3,
    )
    for row in range(len(broken)):
        axes[2].scatter(
            positions + 0.08,
            broken[row],
            color="tab:orange",
            alpha=0.75,
            label="Broken pairing (five seeds)" if row == 0 else None,
        )
    axes[2].set_xticks(positions, ("Max coupled", "Separate + max"))
    axes[2].set_ylim(0.36, 0.60)
    axes[2].set_title("Pairing control")
    axes[2].legend(fontsize=8)

    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def plot_model_comparison(results, output_path=None) -> None:
    """Compare grouped-CV and official-test balanced accuracy."""
    names = list(results)
    cv_scores = [results[name]["grouped_cv"]["balanced_accuracy"] for name in names]
    test_scores = [results[name]["official_test"]["balanced_accuracy"] for name in names]
    intervals = np.asarray([
        results[name]["test_balanced_accuracy_ci"] for name in names
    ])
    test_errors = np.vstack((
        np.asarray(test_scores) - intervals[:, 0],
        intervals[:, 1] - np.asarray(test_scores),
    ))
    positions = np.arange(len(names))
    width = 0.38

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(positions - width / 2, cv_scores, width, label="Nested subject CV")
    axis.bar(
        positions + width / 2,
        test_scores,
        width,
        yerr=test_errors,
        capsize=3,
        label="Official test (subject-bootstrap 95% CI)",
    )
    axis.axhline(1 / 6, color="black", linestyle="--", linewidth=1, label="Chance")
    axis.set_xticks(positions, names, rotation=25, ha="right")
    axis.set_ylabel("Balanced accuracy")
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def plot_confusion_matrix(matrix, output_path=None) -> None:
    """Plot a row-normalized six-activity confusion matrix."""
    names = list(ACTIVITY_NAMES.values())
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    for row in range(6):
        for column in range(6):
            value = matrix[row, column]
            axis.text(
                column, row, f"{value:.2f}", ha="center", va="center",
                color="white" if value > 0.5 else "black",
            )
    axis.set_xticks(range(6), names, rotation=35, ha="right")
    axis.set_yticks(range(6), names)
    axis.set_xlabel("Predicted activity")
    axis.set_ylabel("True activity")
    figure.colorbar(image, ax=axis, label="Recall within true activity")
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def plot_activity_distance_matrix(matrix, output_path=None) -> None:
    """Plot mean stable-rank distances over the sampled interval [0, 2]."""
    names = list(ACTIVITY_NAMES.values())
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="magma")
    for row in range(6):
        for column in range(6):
            axis.text(column, row, f"{matrix[row, column]:.1f}", ha="center", va="center", color="white")
    axis.set_xticks(range(6), names, rotation=35, ha="right")
    axis.set_yticks(range(6), names)
    axis.set_xlabel("Activity")
    axis.set_ylabel("Activity")
    figure.colorbar(image, ax=axis, label="Mean stable-rank $d_{[0,2]}$")
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def plot_sensitivity(results, output_path=None) -> None:
    """Plot grouped-CV scores for embedding and metric perturbations."""
    names = list(results)
    scores = [results[name]["balanced_accuracy"] for name in names]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(names, scores, color="tab:blue")
    axis.axhline(1 / 6, color="black", linestyle="--", linewidth=1)
    axis.set_ylabel("Grouped-CV balanced accuracy")
    axis.set_ylim(0, 0.75)
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def plot_noise_robustness(results, output_path=None) -> None:
    """Plot stable-rank displacement under increasing sensor noise."""
    levels = np.asarray([float(level) for level in results])
    means = np.asarray([results[str(level)]["mean_distance"] for level in levels])
    lows = np.asarray([results[str(level)]["q025"] for level in levels])
    highs = np.asarray([results[str(level)]["q975"] for level in levels])
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.plot(100 * levels, means, marker="o")
    axis.fill_between(100 * levels, lows, highs, alpha=0.2)
    axis.set_xlabel("Noise SD (% of within-window channel SD)")
    axis.set_ylabel("Stable-rank $d_{[0,2]}$ displacement")
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def plot_topology_ablation(results, output_path=None) -> None:
    """Compare the information carried by each sensor and homology degree."""
    names = list(results)
    scores = [results[name]["balanced_accuracy"] for name in names]
    figure, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(names))
    axis.barh(positions, scores, color="tab:blue")
    axis.axvline(1 / 6, color="black", linestyle="--", linewidth=1, label="Chance")
    axis.set_yticks(positions, names)
    axis.set_xlabel("Grouped-CV balanced accuracy")
    axis.set_xlim(0, 0.65)
    axis.invert_yaxis()
    axis.legend()
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def plot_activity_dendrogram(distance_matrix, output_path=None) -> None:
    """Cluster the six subject-balanced activity stable-rank signatures."""
    names = list(ACTIVITY_NAMES.values())
    tree = linkage(squareform(distance_matrix, checks=True), method="average")
    figure, axis = plt.subplots(figsize=(8, 4.8))
    dendrogram(tree, labels=names, leaf_rotation=25, leaf_font_size=9, ax=axis)
    axis.set_ylabel("Stable-rank $d_{[0,2]}$")
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def plot_permutation_null(results, output_path=None) -> None:
    """Plot the subject-wise relabeling null distribution."""
    null_scores = np.asarray(results["null_scores"])
    observed = results["observed_balanced_accuracy"]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(null_scores, bins=30, color="0.7", edgecolor="white")
    axis.axvline(1 / 6, color="black", linestyle="--", label="Chance")
    axis.axvline(observed, color="tab:red", linewidth=2, label=f"Observed = {observed:.3f}")
    axis.set_xlabel("Prototype-level grouped-CV balanced accuracy")
    axis.set_ylabel("Permutations")
    axis.legend()
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def plot_topological_dynamics(results, output_path=None) -> None:
    """Show activity profiles and the strongest interpretable topology effects."""
    names = results["configuration"]["feature_names"]
    means = np.asarray(results["subject_balanced_activity_means"])
    centered = means - means.mean(axis=0)
    spread = means.std(axis=0)
    standardized = np.divide(
        centered,
        spread,
        out=np.zeros_like(centered),
        where=spread > 0,
    )

    short_names = [
        name.replace("Acceleration ", "Acc. ")
        .replace("Gyroscope ", "Gyro. ")
        .replace(" scale", "")
        .replace("Circular ", "Phase ")
        for name in names
    ]
    effects = np.asarray([
        results["activity_partial_eta_squared"][name] for name in names
    ])
    strongest = np.argsort(effects)[-8:]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 5.5),
        gridspec_kw={"width_ratios": (1.8, 1)},
    )
    image = axes[0].imshow(
        standardized,
        aspect="auto",
        cmap="coolwarm",
        vmin=-2,
        vmax=2,
    )
    axes[0].set_xticks(range(len(names)), short_names, rotation=65, ha="right")
    axes[0].set_yticks(range(6), list(ACTIVITY_NAMES.values()))
    axes[0].set_title("Activity profiles")
    figure.colorbar(image, ax=axes[0], label="Standardized activity mean")

    positions = np.arange(len(strongest))
    axes[1].barh(positions, effects[strongest])
    axes[1].set_yticks(positions, [short_names[index] for index in strongest])
    axes[1].set_xlabel(r"Partial $\eta^2$ for activity")
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Strongest activity effects")

    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def plot_dynamics_classification(primary, dynamics, output_path=None) -> None:
    """Compare interpretable topology features with stable ranks and their union."""
    model_results = {
        "Delay stable ranks": primary["models"]["Delay TDA"],
        "Diagram summaries": dynamics["models"]["Diagram summaries"],
        "Circular phase": dynamics["models"]["Circular phase"],
        "Topology + phase": dynamics["models"]["Topological dynamics"],
        "Stable ranks +\ntopology/phase": dynamics["models"][
            "Delay TDA + topological dynamics"
        ],
    }
    names = list(model_results)
    cv_scores = [
        result["grouped_cv"]["balanced_accuracy"]
        for result in model_results.values()
    ]
    test_scores = [
        result["official_test"]["balanced_accuracy"]
        for result in model_results.values()
    ]
    positions = np.arange(len(names))
    width = 0.38

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(positions - width / 2, cv_scores, width, label="Subject-grouped CV")
    axis.bar(positions + width / 2, test_scores, width, label="Official test")
    axis.axhline(1 / 6, color="black", linestyle="--", linewidth=1, label="Chance")
    axis.set_xticks(positions, names, rotation=20, ha="right")
    axis.set_ylabel("Balanced accuracy")
    axis.set_ylim(0, 0.75)
    axis.legend()
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()
