"""Run the complete primary experiment and write reproducible result artifacts."""

import csv
import json
from pathlib import Path

import joblib
import numpy as np

from data_loader import load_engineered_features, load_inertial_data
from embedding import first_nonpositive_lag, mean_autocorrelation
from evaluation import evaluate_feature_set
from features import (
    STABLE_RANK_GRID,
    multi_sensor_stable_ranks,
    stable_rank_l1,
    statistical_features,
)
from topology import persistence_diagrams


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
M = 3
MAX_LAG = 40
SEED = 2956


def _cached_tda(name, train_signals, test_signals, m, lag, **kwargs):
    path = ARTIFACTS / f"{name}_m{m}_lag{lag}.npz"
    if path.exists():
        cached = np.load(path)
        return (
            cached["train_curves"], cached["train_scales"],
            cached["test_curves"], cached["test_scales"],
        )

    print(f"Extracting {name} features", flush=True)
    train_curves, train_scales = multi_sensor_stable_ranks(
        train_signals, m, lag, progress=True, **kwargs
    )
    test_curves, test_scales = multi_sensor_stable_ranks(
        test_signals, m, lag, progress=True, **kwargs
    )
    np.savez_compressed(
        path,
        train_curves=train_curves,
        train_scales=train_scales,
        test_curves=test_curves,
        test_scales=test_scales,
    )
    return train_curves, train_scales, test_curves, test_scales


def _unordered_shuffle_invariance(window):
    shuffled = window[np.random.default_rng(SEED).permutation(len(window))]
    original = persistence_diagrams(window, m=1, lag=1, normalize=True)
    permuted = persistence_diagrams(shuffled, m=1, lag=1, normalize=True)
    return all(
        np.allclose(
            first[np.lexsort((first[:, 1], first[:, 0]))],
            second[np.lexsort((second[:, 1], second[:, 0]))],
            equal_nan=True,
        )
        for first, second in zip(original, permuted)
    )


def _delay_reversal_invariance(window, m, lag):
    """Verify that reversing time gives an isometric Euclidean delay cloud."""
    forward = persistence_diagrams(window, m=m, lag=lag, normalize=True)
    reversed_time = persistence_diagrams(window[::-1], m=m, lag=lag, normalize=True)
    return all(
        np.allclose(
            first[np.lexsort((first[:, 1], first[:, 0]))],
            second[np.lexsort((second[:, 1], second[:, 0]))],
            equal_nan=True,
        )
        for first, second in zip(forward, reversed_time)
    )


def _write_comparison_table(results):
    with (RESULTS / "model_comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "feature_set",
                "nested_cv_balanced_accuracy",
                "test_balanced_accuracy",
                "test_macro_f1",
                "ci_low",
                "ci_high",
            ]
        )
        for name, result in results.items():
            writer.writerow(
                [
                    name,
                    result["grouped_cv"]["balanced_accuracy"],
                    result["official_test"]["balanced_accuracy"],
                    result["official_test"]["macro_f1"],
                    *result["test_balanced_accuracy_ci"],
                ]
            )


def main():
    for directory in (ARTIFACTS, RESULTS):
        directory.mkdir(exist_ok=True)

    train_signals, y_train, train_subjects = load_inertial_data("train")
    test_signals, y_test, test_subjects = load_inertial_data("test")

    autocorrelation = mean_autocorrelation(train_signals["body_acc"], MAX_LAG)
    lag = first_nonpositive_lag(autocorrelation)
    print(f"Selected lag from training autocorrelation: {lag}", flush=True)

    statistics_train = statistical_features(train_signals)
    statistics_test = statistical_features(test_signals)
    uci_train = load_engineered_features("train")
    uci_test = load_engineered_features("test")

    delay_train, delay_scale_train, delay_test, delay_scale_test = _cached_tda(
        "delay_normalized", train_signals, test_signals, M, lag, normalize=True
    )
    instant_train, _, instant_test, _ = _cached_tda(
        "instantaneous_normalized", train_signals, test_signals, 1, 1, normalize=True
    )
    shuffled_train, _, shuffled_test, _ = _cached_tda(
        "shuffled_delay_normalized",
        train_signals,
        test_signals,
        M,
        lag,
        normalize=True,
        shuffle_time=True,
        seed=SEED,
    )

    feature_sets = {
        "Summary statistics": (statistics_train, statistics_test),
        "UCI 561 baseline": (uci_train, uci_test),
        "Instantaneous TDA": (instant_train, instant_test),
        "Delay TDA": (delay_train, delay_test),
        "Delay TDA + scale": (
            np.column_stack((delay_train, delay_scale_train)),
            np.column_stack((delay_test, delay_scale_test)),
        ),
        "Shuffled-delay TDA": (shuffled_train, shuffled_test),
        "Delay TDA + statistics": (
            np.column_stack((delay_train, statistics_train)),
            np.column_stack((delay_test, statistics_test)),
        ),
    }

    model_results = {}
    for name, (X_train, X_test) in feature_sets.items():
        print(f"Evaluating {name}", flush=True)
        result, model = evaluate_feature_set(
            X_train, y_train, train_subjects, X_test, y_test, test_subjects
        )
        model_results[name] = result
        joblib.dump(model, ARTIFACTS / f"model_{name.lower().replace(' ', '_')}.joblib")
        print(
            f"  test balanced accuracy = {result['official_test']['balanced_accuracy']:.3f}",
            flush=True,
        )

    controls = {
        "unordered_shuffle_invariant": _unordered_shuffle_invariance(
            train_signals["body_acc"][0]
        ),
        "delay_time_reversal_invariant": _delay_reversal_invariance(
            train_signals["body_acc"][0], M, lag
        ),
        "mean_delay_vs_shuffled_stable_rank_distance_0_2": float(
            np.mean(stable_rank_l1(delay_train, shuffled_train))
        ),
    }
    output = {
        "configuration": {
            "seed": SEED,
            "embedding_dimension": M,
            "lag": lag,
            "stable_rank_grid": STABLE_RANK_GRID.tolist(),
            "development_validation": "5 outer / 4 inner subject-grouped folds",
            "training_subjects": np.unique(train_subjects).tolist(),
            "test_subjects": np.unique(test_subjects).tolist(),
        },
        "controls": controls,
        "models": model_results,
    }
    (RESULTS / "primary_results.json").write_text(json.dumps(output, indent=2))
    _write_comparison_table(model_results)

    print("Primary analysis complete.", flush=True)


if __name__ == "__main__":
    main()
