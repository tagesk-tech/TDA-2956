"""Run embedding, metric, and noise checks on training subjects."""

import json
from itertools import product
from pathlib import Path

import numpy as np

from data_loader import SENSOR_NAMES, load_inertial_data
from evaluation import fixed_grouped_cv
from features import STABLE_RANK_GRID, multi_sensor_stable_ranks, stable_rank_l1


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
SEED = 2956


def _subject_activity_distances(features, labels, subjects):
    subject_ids = np.unique(subjects)
    prototypes = {
        (subject, activity): features[(subjects == subject) & (labels == activity)].mean(axis=0)
        for subject in subject_ids
        for activity in range(1, 7)
    }
    matrix = np.zeros((6, 6))

    for first_activity in range(1, 7):
        for second_activity in range(1, 7):
            distances = [
                stable_rank_l1(
                    prototypes[first_subject, first_activity],
                    prototypes[second_subject, second_activity],
                )
                for first_subject in subject_ids
                for second_subject in subject_ids
                if first_subject != second_subject
            ]
            matrix[first_activity - 1, second_activity - 1] = np.mean(distances)
    return matrix


def _paired_subject_difference(first_predictions, second_predictions, labels, subjects, repeats=10000):
    subject_ids = np.unique(subjects)
    differences = []
    for subject in subject_ids:
        mask = subjects == subject
        first = np.mean([
            np.mean(first_predictions[mask & (labels == activity)] == activity)
            for activity in range(1, 7)
        ])
        second = np.mean([
            np.mean(second_predictions[mask & (labels == activity)] == activity)
            for activity in range(1, 7)
        ])
        differences.append(first - second)

    differences = np.asarray(differences)
    rng = np.random.default_rng(SEED)
    bootstraps = np.mean(
        rng.choice(differences, size=(repeats, len(differences)), replace=True), axis=1
    )
    observed = abs(differences.mean())
    signed_means = np.asarray([
        np.mean(differences * signs)
        for signs in product((-1.0, 1.0), repeat=len(differences))
    ])
    return {
        "mean": float(differences.mean()),
        "ci": np.quantile(bootstraps, (0.025, 0.975)).tolist(),
        "exact_two_sided_sign_flip_p": float(
            np.mean(np.abs(signed_means) >= observed - 1e-15)
        ),
        "subject_differences": differences.tolist(),
    }


def _cached_sensitivity(signals, m, lag, metric="euclidean", grid=STABLE_RANK_GRID):
    path = ARTIFACTS / f"sensitivity_{metric}_m{m}_lag{lag}.npz"
    if path.exists():
        return np.load(path)["features"]
    features, _ = multi_sensor_stable_ranks(
        signals,
        m,
        lag,
        normalize=True,
        metric=metric,
        grid=grid,
        progress=True,
    )
    np.savez_compressed(path, features=features)
    return features


def _noise_experiment(signals, reference_features, labels, m, lag):
    rng = np.random.default_rng(SEED)
    selected = np.concatenate(
        [rng.choice(np.flatnonzero(labels == activity), 60, replace=False) for activity in range(1, 7)]
    )
    base_noise = {
        sensor: rng.normal(size=signals[sensor][selected].shape) for sensor in SENSOR_NAMES
    }
    results = {}

    for level in (0.02, 0.05, 0.10):
        noisy = {}
        for sensor in SENSOR_NAMES:
            windows = signals[sensor][selected]
            channel_sd = windows.std(axis=1, keepdims=True)
            noisy[sensor] = windows + level * channel_sd * base_noise[sensor]
        features, _ = multi_sensor_stable_ranks(
            noisy, m, lag, normalize=True, progress=False
        )
        distances = stable_rank_l1(features, reference_features[selected])
        results[str(level)] = {
            "mean_distance": float(distances.mean()),
            "q025": float(np.quantile(distances, 0.025)),
            "q975": float(np.quantile(distances, 0.975)),
        }
    return results


def main():
    signals, labels, subjects = load_inertial_data("train")
    _, test_labels, test_subjects = load_inertial_data("test")
    primary = json.loads((RESULTS / "primary_results.json").read_text())
    configuration = primary["configuration"]
    primary_m = configuration["embedding_dimension"]
    primary_lag = configuration["lag"]
    delay_features = np.load(
        ARTIFACTS / f"delay_normalized_m{primary_m}_lag{primary_lag}.npz"
    )["train_curves"]
    parameters = primary["models"]["Delay TDA"]["parameters"]

    distance_matrix = _subject_activity_distances(delay_features, labels, subjects)
    diagonal = np.diag(distance_matrix)
    between = distance_matrix[~np.eye(6, dtype=bool)]

    sensitivity = {}
    for m, lag in ((2, 15), (3, 8), (3, 15), (3, 22), (4, 15)):
        print(f"Sensitivity: m={m}, lag={lag}", flush=True)
        features = delay_features if (m, lag) == (3, 15) else _cached_sensitivity(signals, m, lag)
        sensitivity[f"m={m}, lag={lag}"] = fixed_grouped_cv(
            features,
            labels,
            subjects,
            C=parameters["svc__C"],
            gamma=parameters["svc__gamma"],
        )

    print("Sensitivity: cityblock metric", flush=True)
    cityblock_grid = np.linspace(0.0, 4.0, 64)
    cityblock = _cached_sensitivity(
        signals, 3, 15, metric="cityblock", grid=cityblock_grid
    )
    sensitivity["cityblock, m=3, lag=15"] = fixed_grouped_cv(
        cityblock,
        labels,
        subjects,
        C=parameters["svc__C"],
        gamma=parameters["svc__gamma"],
    )

    models = primary["models"]
    paired_controls = {
        "delay_minus_instantaneous": _paired_subject_difference(
            np.asarray(models["Delay TDA"]["test_predictions"]),
            np.asarray(models["Instantaneous TDA"]["test_predictions"]),
            test_labels,
            test_subjects,
        ),
        "delay_minus_shuffled": _paired_subject_difference(
            np.asarray(models["Delay TDA"]["test_predictions"]),
            np.asarray(models["Shuffled-delay TDA"]["test_predictions"]),
            test_labels,
            test_subjects,
        ),
    }
    noise = _noise_experiment(
        signals, delay_features, labels, primary_m, primary_lag
    )

    output = {
        "subject_activity_distances": distance_matrix.tolist(),
        "mean_within_activity_distance": float(diagonal.mean()),
        "mean_between_activity_distance": float(between.mean()),
        "between_within_ratio": float(between.mean() / diagonal.mean()),
        "paired_controls": paired_controls,
        "sensitivity": sensitivity,
        "noise": noise,
    }
    (RESULTS / "robustness_results.json").write_text(json.dumps(output, indent=2))
    print("Robustness analysis complete.", flush=True)


if __name__ == "__main__":
    main()
