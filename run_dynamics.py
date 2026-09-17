"""Interpret persistent loops as activity dynamics and test them in classification."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr

from data_loader import load_inertial_data
from evaluation import evaluate_feature_set
from features import (
    multi_sensor_topological_dynamics,
    topological_dynamics_feature_names,
)
from run_extensions import paired_subject_bootstrap_difference


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
M = 3
LAG = 15


def _cached_features(train_signals, test_signals):
    path = ARTIFACTS / f"topological_dynamics_m{M}_lag{LAG}.npz"
    if path.exists():
        cache = np.load(path)
        return (
            cache["train_features"],
            cache["train_scales"],
            cache["test_features"],
            cache["test_scales"],
        )

    print("Extracting topology and circular-phase features", flush=True)
    train_features, train_scales = multi_sensor_topological_dynamics(
        train_signals, M, LAG, progress=True
    )
    test_features, test_scales = multi_sensor_topological_dynamics(
        test_signals, M, LAG, progress=True
    )
    np.savez_compressed(
        path,
        train_features=train_features,
        train_scales=train_scales,
        test_features=test_features,
        test_scales=test_scales,
    )
    return train_features, train_scales, test_features, test_scales


def _subject_activity_values(features, labels, subjects):
    """Return a subjects-by-activities-by-features array of window means."""
    values = []
    for subject in np.unique(subjects):
        values.append([
            features[(subjects == subject) & (labels == activity)].mean(axis=0)
            for activity in range(1, 7)
        ])
    return np.asarray(values)


def _activity_effect(values):
    """Partial eta squared for activity after removing each subject mean."""
    grand_mean = values.mean(axis=(0, 1))
    activity_means = values.mean(axis=0)
    subject_means = values.mean(axis=1)
    residual = (
        values
        - subject_means[:, None, :]
        - activity_means[None, :, :]
        + grand_mean[None, None, :]
    )
    activity_sum_squares = len(values) * np.sum(
        (activity_means - grand_mean) ** 2, axis=0
    )
    error_sum_squares = np.sum(residual**2, axis=(0, 1))
    return activity_sum_squares / (activity_sum_squares + error_sum_squares)


def _dominant_frequency(windows, sampling_rate=50.0):
    centered = windows - windows.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    power = power.sum(axis=2)
    power[:, 0] = 0
    frequencies = np.fft.rfftfreq(windows.shape[1], d=1.0 / sampling_rate)
    return frequencies[np.argmax(power, axis=1)]


def _frequency_validation(signals, labels, features):
    """Compare circular-coordinate frequency with ordinary spectral frequency."""
    output = {}
    for sensor, feature_column in (("body_acc", 6), ("body_gyro", 14)):
        spectral = _dominant_frequency(signals[sensor])
        circular = features[:, feature_column]
        sensor_results = {}
        for group, mask in (
            ("all", np.ones(len(labels), dtype=bool)),
            ("dynamic", labels <= 3),
            ("static", labels >= 4),
        ):
            correlation, p_value = spearmanr(circular[mask], spectral[mask])
            sensor_results[group] = {
                "spearman_correlation": float(correlation),
                "p_value": float(p_value),
                "mean_circular_frequency": float(circular[mask].mean()),
                "mean_spectral_frequency": float(spectral[mask].mean()),
            }
        output[sensor] = sensor_results
    return output


def _evaluate(name, train, labels, subjects, test, test_labels, test_subjects):
    print(f"Evaluating {name}", flush=True)
    result, model = evaluate_feature_set(
        train,
        labels,
        subjects,
        test,
        test_labels,
        test_subjects,
    )
    filename = name.lower().replace(" ", "_").replace("+", "plus")
    joblib.dump(model, ARTIFACTS / f"model_{filename}.joblib")
    print(
        "  grouped CV / test = "
        f"{result['grouped_cv']['balanced_accuracy']:.3f} / "
        f"{result['official_test']['balanced_accuracy']:.3f}",
        flush=True,
    )
    return result


def main(*, summarize_only=False):
    ARTIFACTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    train_signals, labels, subjects = load_inertial_data("train")
    test_signals, test_labels, test_subjects = load_inertial_data("test")
    train, _, test, _ = _cached_features(train_signals, test_signals)

    stable_rank_cache = np.load(ARTIFACTS / "delay_normalized_m3_lag15.npz")
    stable_rank_train = stable_rank_cache["train_curves"]
    stable_rank_test = stable_rank_cache["test_curves"]

    block_width = train.shape[1] // 2
    diagram_columns = np.r_[0:6, block_width : block_width + 6]
    phase_columns = np.r_[6:8, block_width + 6 : block_width + 8]
    feature_sets = {
        "Diagram summaries": (train[:, diagram_columns], test[:, diagram_columns]),
        "Circular phase": (train[:, phase_columns], test[:, phase_columns]),
        "Topological dynamics": (train, test),
        "Delay TDA + topological dynamics": (
            np.column_stack((stable_rank_train, train)),
            np.column_stack((stable_rank_test, test)),
        ),
    }
    result_path = RESULTS / "dynamics_results.json"
    if summarize_only:
        model_results = json.loads(result_path.read_text())["models"]
    else:
        model_results = {
            name: _evaluate(
                name,
                train_features,
                labels,
                subjects,
                test_features,
                test_labels,
                test_subjects,
            )
            for name, (train_features, test_features) in feature_sets.items()
        }

    subject_activity = _subject_activity_values(train, labels, subjects)
    means = subject_activity.mean(axis=0)
    standard_errors = subject_activity.std(axis=0, ddof=1) / np.sqrt(
        len(subject_activity)
    )
    feature_names = topological_dynamics_feature_names()
    effect_sizes = _activity_effect(subject_activity)
    primary = json.loads((RESULTS / "primary_results.json").read_text())
    delay_predictions = np.asarray(
        primary["models"]["Delay TDA"]["test_predictions"]
    )
    dynamics_predictions = np.asarray(
        model_results["Topological dynamics"]["test_predictions"]
    )
    combined_predictions = np.asarray(
        model_results["Delay TDA + topological dynamics"]["test_predictions"]
    )

    output = {
        "protocol": (
            "The same features are computed for every window and activity. "
            "Models use nested subject-grouped development folds and the "
            "subject-disjoint official test. This branch is post-hoc."
        ),
        "configuration": {
            "embedding_dimension": M,
            "lag": LAG,
            "sampling_rate": 50.0,
            "feature_names": list(feature_names),
            "diagram_summary_columns": diagram_columns.tolist(),
            "circular_phase_columns": phase_columns.tolist(),
        },
        "subject_balanced_activity_means": means.tolist(),
        "subject_balanced_activity_standard_errors": standard_errors.tolist(),
        "activity_partial_eta_squared": {
            name: float(value) for name, value in zip(feature_names, effect_sizes)
        },
        "frequency_validation": _frequency_validation(
            train_signals, labels, train
        ),
        "test_comparisons": {
            "dynamics_minus_delay_tda": paired_subject_bootstrap_difference(
                dynamics_predictions,
                delay_predictions,
                test_labels,
                test_subjects,
            ),
            "combined_minus_dynamics": paired_subject_bootstrap_difference(
                combined_predictions,
                dynamics_predictions,
                test_labels,
                test_subjects,
            ),
        },
        "models": model_results,
    }
    result_path.write_text(json.dumps(output, indent=2))
    print("Topological dynamics analysis complete.", flush=True)


if __name__ == "__main__":
    main(summarize_only="--summarize-only" in sys.argv)
