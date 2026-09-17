"""Regenerate every report figure from cached experiment outputs."""

import json
from pathlib import Path

import numpy as np

from data_loader import load_inertial_data
from embedding import first_nonpositive_lag, mean_autocorrelation
from plotting import (
    plot_activity_dendrogram,
    plot_activity_distance_matrix,
    plot_activity_persistence,
    plot_confusion_matrix,
    plot_coupled_max_analysis,
    plot_dynamics_classification,
    plot_mean_autocorrelation,
    plot_model_comparison,
    plot_noise_robustness,
    plot_joint_analysis,
    plot_joint_stable_rank_means,
    plot_permutation_null,
    plot_sensitivity,
    plot_stable_rank_means,
    plot_synchronization_control,
    plot_topology_ablation,
    plot_topological_dynamics,
)
from topology import representative_diagrams


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"


def main():
    FIGURES.mkdir(exist_ok=True)
    signals, labels, subjects = load_inertial_data("train")
    primary = json.loads((ROOT / "results" / "primary_results.json").read_text())
    robustness = json.loads((ROOT / "results" / "robustness_results.json").read_text())
    extensions = json.loads((ROOT / "results" / "extension_results.json").read_text())
    dynamics = json.loads((ROOT / "results" / "dynamics_results.json").read_text())

    configuration = primary["configuration"]
    m = configuration["embedding_dimension"]
    lag = configuration["lag"]
    stable_rank_grid = np.asarray(configuration["stable_rank_grid"])
    delay_features = np.load(
        ROOT / "artifacts" / f"delay_normalized_m{m}_lag{lag}.npz"
    )["train_curves"]
    joint_features = np.load(
        ROOT / "artifacts" / f"joint_normalized_w50_m{m}_lag{lag}.npz"
    )["features"]
    if delay_features.shape[1] != 4 * len(stable_rank_grid):
        raise ValueError("cached feature width does not match the saved stable-rank grid")

    autocorrelation = mean_autocorrelation(signals["body_acc"], 40)
    chosen_lag = first_nonpositive_lag(autocorrelation)
    if chosen_lag != lag:
        raise ValueError(
            f"saved analysis uses lag {lag}, but the training data now gives {chosen_lag}"
        )
    plot_mean_autocorrelation(
        autocorrelation, chosen_lag, FIGURES / "autocorrelation_lag.png"
    )

    diagrams = representative_diagrams(
        signals["body_acc"],
        labels,
        subjects,
        subject=1,
        m=m,
        lag=lag,
        normalize=True,
    )
    plot_activity_persistence(diagrams, FIGURES / "representative_persistence.png")

    for model_name, filename in (
        ("Delay TDA", "confusion_delay_tda.png"),
        ("Summary statistics", "confusion_summary_statistics.png"),
        ("UCI 561 baseline", "confusion_uci_baseline.png"),
    ):
        matrix = np.asarray(primary["models"][model_name]["test_confusion_matrix"])
        plot_confusion_matrix(matrix, FIGURES / filename)

    plot_stable_rank_means(
        delay_features,
        labels,
        subjects,
        stable_rank_grid,
        FIGURES / "stable_rank_means.png",
    )
    plot_joint_stable_rank_means(
        joint_features,
        labels,
        subjects,
        stable_rank_grid,
        FIGURES / "joint_stable_rank_means.png",
    )
    plot_joint_analysis(
        extensions["joint_representation"],
        FIGURES / "joint_representation.png",
    )
    plot_synchronization_control(
        extensions["joint_representation"],
        FIGURES / "synchronization_control.png",
    )
    plot_coupled_max_analysis(
        extensions["coupled_max_representation"],
        FIGURES / "coupled_max_filtration.png",
    )
    plot_model_comparison(primary["models"], FIGURES / "model_comparison.png")
    plot_activity_distance_matrix(
        np.asarray(robustness["subject_activity_distances"]),
        FIGURES / "activity_stable_rank_distances.png",
    )
    plot_sensitivity(
        robustness["sensitivity"], FIGURES / "parameter_sensitivity.png"
    )
    plot_noise_robustness(
        robustness["noise"], FIGURES / "noise_robustness.png"
    )
    plot_topology_ablation(
        extensions["ablations"], FIGURES / "topology_ablation.png"
    )
    plot_activity_dendrogram(
        np.asarray(extensions["prototype_activity_distance_matrix"]),
        FIGURES / "activity_dendrogram.png",
    )
    plot_permutation_null(
        extensions["prototype_permutation_test"],
        FIGURES / "prototype_permutation_null.png",
    )
    plot_topological_dynamics(
        dynamics,
        FIGURES / "topological_dynamics.png",
    )
    plot_dynamics_classification(
        primary,
        dynamics,
        FIGURES / "dynamics_classification.png",
    )


if __name__ == "__main__":
    main()
