"""Experiment functions called explicitly by the analysis notebook."""

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr

from data_loader import SENSOR_NAMES, load_engineered_features
from embedding import first_nonpositive_lag, mean_autocorrelation
from evaluation import (
    activity_distance_matrix,
    activity_effect,
    classification_metrics,
    cross_subject_activity_distances,
    evaluate_feature_set,
    fixed_grouped_cv,
    fixed_grouped_predictions,
    paired_subject_bootstrap_difference,
    paired_subject_comparison,
    prototype_classifier,
    prototype_permutation_test,
    subject_activity_prototypes,
)
from features import (
    STABLE_RANK_GRID,
    coupled_sensor_stable_ranks,
    joint_sensor_stable_ranks,
    multi_sensor_stable_ranks,
    multi_sensor_topological_dynamics,
    stable_rank_blocks,
    stable_rank_l1,
    statistical_features,
    topological_dynamics_feature_names,
)
from topology import persistence_diagrams


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
SEED = 2956
M = 3
MAX_LAG = 40
__all__ = (
    "primary_analysis",
    "robustness_analysis",
    "extension_analysis",
    "dynamics_analysis",
)


# Primary stable-rank experiment -------------------------------------------------


def _cached_tda(name, train_signals, test_signals, m, lag, **kwargs):
    path = ARTIFACTS / f"{name}_m{m}_lag{lag}.npz"
    if path.exists():
        cached = np.load(path)
        return (
            cached["train_curves"],
            cached["train_scales"],
            cached["test_curves"],
            cached["test_scales"],
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
    forward = persistence_diagrams(window, m=m, lag=lag, normalize=True)
    reversed_time = persistence_diagrams(window[::-1], m=m, lag=lag, normalize=True)
    return all(
        np.allclose(
            first[np.lexsort((first[:, 1], first[:, 0]))],
            second[np.lexsort((second[:, 1], second[:, 0]))],
            equal_nan=True,
            atol=1e-6,
        )
        for first, second in zip(forward, reversed_time)
    )


def _write_comparison_table(results):
    with (RESULTS / "model_comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "representation",
            "nested_cv_balanced_accuracy",
            "test_balanced_accuracy",
            "test_macro_f1",
            "test_ci_lower",
            "test_ci_upper",
        ])
        for name, result in results.items():
            writer.writerow([
                name,
                result["grouped_cv"]["balanced_accuracy"],
                result["official_test"]["balanced_accuracy"],
                result["official_test"]["macro_f1"],
                *result["test_balanced_accuracy_ci"],
            ])


def primary_analysis(
    train_signals,
    train_labels,
    train_subjects,
    test_signals,
    test_labels,
    test_subjects,
    *,
    recompute=False,
):
    """Run or load the seven primary feature comparisons."""
    ARTIFACTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    autocorrelation = mean_autocorrelation(train_signals["body_acc"], MAX_LAG)
    lag = first_nonpositive_lag(autocorrelation)

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
    result_path = RESULTS / "primary_results.json"
    if result_path.exists() and not recompute:
        output = json.loads(result_path.read_text())
    else:
        model_results = {}
        for name, (X_train, X_test) in feature_sets.items():
            print(f"Evaluating {name}", flush=True)
            result, model = evaluate_feature_set(
                X_train,
                train_labels,
                train_subjects,
                X_test,
                test_labels,
                test_subjects,
            )
            model_results[name] = result
            filename = f"model_{name.lower().replace(' ', '_')}.joblib"
            joblib.dump(model, ARTIFACTS / filename)
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
            "controls": {
                "unordered_shuffle_invariant": _unordered_shuffle_invariance(
                    train_signals["body_acc"][0]
                ),
                "delay_time_reversal_invariant": _delay_reversal_invariance(
                    train_signals["body_acc"][0], M, lag
                ),
                "mean_delay_vs_shuffled_stable_rank_distance_0_2": float(
                    np.mean(stable_rank_l1(delay_train, shuffled_train))
                ),
            },
            "models": model_results,
        }
        result_path.write_text(json.dumps(output, indent=2))
        _write_comparison_table(model_results)

    arrays = {
        "delay_train": delay_train,
        "delay_test": delay_test,
        "instant_train": instant_train,
        "instant_test": instant_test,
        "shuffled_train": shuffled_train,
        "shuffled_test": shuffled_test,
        "delay_scale_train": delay_scale_train,
        "delay_scale_test": delay_scale_test,
    }
    return output, arrays


# Robustness and temporal controls ----------------------------------------------


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
    selected = np.concatenate([
        rng.choice(np.flatnonzero(labels == activity), 60, replace=False)
        for activity in range(1, 7)
    ])
    base_noise = {
        sensor: rng.normal(size=signals[sensor][selected].shape)
        for sensor in SENSOR_NAMES
    }
    output = {}
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
        output[str(level)] = {
            "mean_distance": float(distances.mean()),
            "q025": float(np.quantile(distances, 0.025)),
            "q975": float(np.quantile(distances, 0.975)),
        }
    return output


def robustness_analysis(
    train_signals,
    train_labels,
    train_subjects,
    test_labels,
    test_subjects,
    primary,
    primary_features,
    *,
    recompute=False,
):
    """Run or load embedding, metric, temporal-control, and noise checks."""
    result_path = RESULTS / "robustness_results.json"
    if result_path.exists() and not recompute:
        return json.loads(result_path.read_text())

    m = primary["configuration"]["embedding_dimension"]
    lag = primary["configuration"]["lag"]
    delay_features = primary_features["delay_train"]
    parameters = primary["models"]["Delay TDA"]["parameters"]
    distance_matrix = cross_subject_activity_distances(
        delay_features, train_labels, train_subjects
    )
    diagonal = np.diag(distance_matrix)
    between = distance_matrix[~np.eye(6, dtype=bool)]

    sensitivity = {}
    for candidate_m, candidate_lag in ((2, 15), (3, 8), (3, 15), (3, 22), (4, 15)):
        features = (
            delay_features
            if (candidate_m, candidate_lag) == (m, lag)
            else _cached_sensitivity(train_signals, candidate_m, candidate_lag)
        )
        sensitivity[f"m={candidate_m}, lag={candidate_lag}"] = fixed_grouped_cv(
            features,
            train_labels,
            train_subjects,
            C=parameters["svc__C"],
            gamma=parameters["svc__gamma"],
        )
    cityblock_grid = np.linspace(0.0, 4.0, 64)
    cityblock = _cached_sensitivity(
        train_signals, m, lag, metric="cityblock", grid=cityblock_grid
    )
    sensitivity[f"cityblock, m={m}, lag={lag}"] = fixed_grouped_cv(
        cityblock,
        train_labels,
        train_subjects,
        C=parameters["svc__C"],
        gamma=parameters["svc__gamma"],
    )
    models = primary["models"]
    output = {
        "subject_activity_distances": distance_matrix.tolist(),
        "mean_within_activity_distance": float(diagonal.mean()),
        "mean_between_activity_distance": float(between.mean()),
        "between_within_ratio": float(between.mean() / diagonal.mean()),
        "paired_controls": {
            "delay_minus_instantaneous": paired_subject_comparison(
                np.asarray(models["Delay TDA"]["test_predictions"]),
                np.asarray(models["Instantaneous TDA"]["test_predictions"]),
                test_labels,
                test_subjects,
            ),
            "delay_minus_shuffled": paired_subject_comparison(
                np.asarray(models["Delay TDA"]["test_predictions"]),
                np.asarray(models["Shuffled-delay TDA"]["test_predictions"]),
                test_labels,
                test_subjects,
            ),
        },
        "sensitivity": sensitivity,
        "noise": _noise_experiment(
            train_signals, delay_features, train_labels, m, lag
        ),
    }
    result_path.write_text(json.dumps(output, indent=2))
    return output


# Circular-coordinate dynamics --------------------------------------------------


def _cached_dynamics(train_signals, test_signals, m, lag):
    path = ARTIFACTS / f"topological_dynamics_m{m}_lag{lag}.npz"
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
        train_signals, m, lag, progress=True
    )
    test_features, test_scales = multi_sensor_topological_dynamics(
        test_signals, m, lag, progress=True
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
    values = []
    for subject in np.unique(subjects):
        values.append([
            features[(subjects == subject) & (labels == activity)].mean(axis=0)
            for activity in range(1, 7)
        ])
    return np.asarray(values)


def _dominant_frequency(windows, sampling_rate=50.0):
    centered = windows - windows.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    power = power.sum(axis=2)
    power[:, 0] = 0
    frequencies = np.fft.rfftfreq(windows.shape[1], d=1.0 / sampling_rate)
    return frequencies[np.argmax(power, axis=1)]


def _frequency_validation(signals, labels, features):
    output = {}
    for sensor, feature_column in (("body_acc", 6), ("body_gyro", 14)):
        spectral = _dominant_frequency(signals[sensor])
        circular = features[:, feature_column]
        sensor_output = {}
        for group, mask in (
            ("all", np.ones(len(labels), dtype=bool)),
            ("dynamic", labels <= 3),
            ("static", labels >= 4),
        ):
            correlation, p_value = spearmanr(circular[mask], spectral[mask])
            sensor_output[group] = {
                "spearman_correlation": float(correlation),
                "p_value": float(p_value),
                "mean_circular_frequency": float(circular[mask].mean()),
                "mean_spectral_frequency": float(spectral[mask].mean()),
            }
        output[sensor] = sensor_output
    return output


def dynamics_analysis(
    train_signals,
    train_labels,
    train_subjects,
    test_signals,
    test_labels,
    test_subjects,
    primary,
    primary_features,
    *,
    recompute=False,
):
    """Run or load the all-activity topology-and-circular-phase analysis."""
    m = primary["configuration"]["embedding_dimension"]
    lag = primary["configuration"]["lag"]
    train, _, test, _ = _cached_dynamics(train_signals, test_signals, m, lag)
    block_width = train.shape[1] // 2
    diagram_columns = np.r_[0:6, block_width : block_width + 6]
    phase_columns = np.r_[6:8, block_width + 6 : block_width + 8]
    feature_sets = {
        "Diagram summaries": (train[:, diagram_columns], test[:, diagram_columns]),
        "Circular phase": (train[:, phase_columns], test[:, phase_columns]),
        "Topological dynamics": (train, test),
        "Delay TDA + topological dynamics": (
            np.column_stack((primary_features["delay_train"], train)),
            np.column_stack((primary_features["delay_test"], test)),
        ),
    }
    result_path = RESULTS / "dynamics_results.json"
    if result_path.exists() and not recompute:
        return json.loads(result_path.read_text()), train

    model_results = {}
    for name, (X_train, X_test) in feature_sets.items():
        print(f"Evaluating {name}", flush=True)
        result, model = evaluate_feature_set(
            X_train,
            train_labels,
            train_subjects,
            X_test,
            test_labels,
            test_subjects,
        )
        model_results[name] = result
        filename = f"model_{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, ARTIFACTS / filename)

    subject_activity = _subject_activity_values(train, train_labels, train_subjects)
    means = subject_activity.mean(axis=0)
    standard_errors = subject_activity.std(axis=0, ddof=1) / np.sqrt(
        len(subject_activity)
    )
    feature_names = topological_dynamics_feature_names()
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
            "embedding_dimension": m,
            "lag": lag,
            "sampling_rate": 50.0,
            "feature_names": list(feature_names),
            "diagram_summary_columns": diagram_columns.tolist(),
            "circular_phase_columns": phase_columns.tolist(),
        },
        "subject_balanced_activity_means": means.tolist(),
        "subject_balanced_activity_standard_errors": standard_errors.tolist(),
        "activity_partial_eta_squared": {
            name: float(value)
            for name, value in zip(feature_names, activity_effect(subject_activity))
        },
        "frequency_validation": _frequency_validation(
            train_signals, train_labels, train
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
    return output, train


# Coupling, ablations, clustering, and prototype tests --------------------------


def _cached_joint_features(
    signals,
    m,
    lag,
    acceleration_weight,
    *,
    desynchronize=False,
    seed=SEED,
):
    weight_code = int(round(100 * acceleration_weight))
    if desynchronize:
        filename = (
            f"joint_desynchronized_w{weight_code}_m{m}_lag{lag}_seed{seed}.npz"
        )
    else:
        filename = f"joint_normalized_w{weight_code}_m{m}_lag{lag}.npz"
    path = ARTIFACTS / filename
    if path.exists():
        return np.load(path)["features"]

    description = "desynchronized" if desynchronize else "synchronized"
    print(f"Joint topology: {description}, seed {seed}", flush=True)
    features, scales = joint_sensor_stable_ranks(
        signals,
        m,
        lag,
        acceleration_weight=acceleration_weight,
        desynchronize=desynchronize,
        seed=seed,
        progress=True,
    )
    np.savez_compressed(path, features=features, scales=scales)
    return features


def _cached_coupled_features(
    signals,
    m,
    lag,
    acceleration_weight,
    *,
    desynchronize=False,
    seed=SEED,
):
    """Load or compute stable ranks for a max-coupled sensor filtration."""
    weight_code = int(round(100 * acceleration_weight))
    pairing = f"desynchronized_seed{seed}" if desynchronize else "synchronized"
    path = ARTIFACTS / f"coupled_max_{pairing}_w{weight_code}_m{m}_lag{lag}.npz"
    if path.exists():
        return np.load(path)["features"]

    print(f"Max-coupled topology: {pairing}, weight {acceleration_weight:.2f}", flush=True)
    features, scales = coupled_sensor_stable_ranks(
        signals,
        m,
        lag,
        acceleration_weight=acceleration_weight,
        desynchronize=desynchronize,
        seed=seed,
        progress=True,
    )
    np.savez_compressed(path, features=features, scales=scales)
    return features


def extension_analysis(
    train_signals,
    train_labels,
    train_subjects,
    test_labels,
    test_subjects,
    primary,
    primary_features,
    *,
    recompute=False,
):
    """Run or load the post-hoc ablation, prototype, and coupling analyses."""
    result_path = RESULTS / "extension_results.json"
    if result_path.exists() and not recompute:
        _cached_joint_features(
            train_signals,
            primary["configuration"]["embedding_dimension"],
            primary["configuration"]["lag"],
            0.50,
        )
        return json.loads(result_path.read_text())

    train_features = primary_features["delay_train"]
    test_features = primary_features["delay_test"]

    print("Homology and sensor ablations", flush=True)
    parameters = primary["models"]["Delay TDA"]["parameters"]
    ablations = {
        name: fixed_grouped_cv(
            block,
            train_labels,
            train_subjects,
            C=parameters["svc__C"],
            gamma=parameters["svc__gamma"],
        )
        for name, block in stable_rank_blocks(train_features).items()
    }

    print("Joint acceleration-gyroscope topology", flush=True)
    joint_by_weight = {
        weight: _cached_joint_features(train_signals, 3, 15, weight)
        for weight in (0.25, 0.50, 0.75)
    }
    joint_features = joint_by_weight[0.50]
    desynchronization_seeds = tuple(range(SEED, SEED + 5))
    desynchronized_by_seed = {
        seed: _cached_joint_features(
            train_signals,
            3,
            15,
            0.50,
            desynchronize=True,
            seed=seed,
        )
        for seed in desynchronization_seeds
    }
    primary_desynchronized = desynchronized_by_seed[SEED]
    joint_feature_sets = {
        "Separate sensor topology": train_features,
        "Joint sensor topology": joint_features,
        "Desynchronized joint topology": primary_desynchronized,
        "Separate + joint topology": np.column_stack((train_features, joint_features)),
        "Separate + desynchronized joint topology": np.column_stack(
            (train_features, primary_desynchronized)
        ),
    }
    joint_predictions = {
        name: fixed_grouped_predictions(
            features,
            train_labels,
            train_subjects,
            C=parameters["svc__C"],
            gamma=parameters["svc__gamma"],
        )
        for name, features in joint_feature_sets.items()
    }
    joint_scores = {
        name: classification_metrics(train_labels, predictions)
        for name, predictions in joint_predictions.items()
    }
    joint_scores.update({
        "Joint H0 only": fixed_grouped_cv(
            joint_features[:, : len(STABLE_RANK_GRID)],
            train_labels,
            train_subjects,
            C=parameters["svc__C"],
            gamma=parameters["svc__gamma"],
        ),
        "Joint H1 only": fixed_grouped_cv(
            joint_features[:, len(STABLE_RANK_GRID) :],
            train_labels,
            train_subjects,
            C=parameters["svc__C"],
            gamma=parameters["svc__gamma"],
        ),
    })
    combined_minus_separate = paired_subject_bootstrap_difference(
        joint_predictions["Separate + joint topology"],
        joint_predictions["Separate sensor topology"],
        train_labels,
        train_subjects,
    )
    synchronized_minus_desynchronized = paired_subject_bootstrap_difference(
        joint_predictions["Joint sensor topology"],
        joint_predictions["Desynchronized joint topology"],
        train_labels,
        train_subjects,
    )
    combined_synchronized_minus_desynchronized = (
        paired_subject_bootstrap_difference(
            joint_predictions["Separate + joint topology"],
            joint_predictions["Separate + desynchronized joint topology"],
            train_labels,
            train_subjects,
        )
    )
    desynchronized_combined_minus_separate = paired_subject_bootstrap_difference(
        joint_predictions["Separate + desynchronized joint topology"],
        joint_predictions["Separate sensor topology"],
        train_labels,
        train_subjects,
    )
    joint_weight_sensitivity = {
        f"acceleration weight {weight:.2f}": fixed_grouped_cv(
            features,
            train_labels,
            train_subjects,
            C=parameters["svc__C"],
            gamma=parameters["svc__gamma"],
        )
        for weight, features in joint_by_weight.items()
    }
    desynchronization_sensitivity = {}
    for seed, features in desynchronized_by_seed.items():
        if seed == SEED:
            joint_desynchronized_predictions = joint_predictions[
                "Desynchronized joint topology"
            ]
            combined_desynchronized_predictions = joint_predictions[
                "Separate + desynchronized joint topology"
            ]
        else:
            joint_desynchronized_predictions = fixed_grouped_predictions(
                features,
                train_labels,
                train_subjects,
                C=parameters["svc__C"],
                gamma=parameters["svc__gamma"],
            )
            combined_desynchronized_predictions = fixed_grouped_predictions(
                np.column_stack((train_features, features)),
                train_labels,
                train_subjects,
                C=parameters["svc__C"],
                gamma=parameters["svc__gamma"],
            )
        desynchronization_sensitivity[str(seed)] = {
            "joint": classification_metrics(train_labels, joint_desynchronized_predictions),
            "separate_plus_joint": classification_metrics(
                train_labels, combined_desynchronized_predictions
            ),
        }

    print("Max-coupled acceleration-gyroscope filtration", flush=True)
    coupled_by_weight = {
        weight: _cached_coupled_features(train_signals, 3, 15, weight)
        for weight in (0.25, 0.50, 0.75)
    }
    coupled_features = coupled_by_weight[0.50]
    coupled_desynchronized_by_seed = {
        seed: _cached_coupled_features(
            train_signals,
            3,
            15,
            0.50,
            desynchronize=True,
            seed=seed,
        )
        for seed in desynchronization_seeds
    }
    coupled_primary_desynchronized = coupled_desynchronized_by_seed[SEED]
    coupled_feature_sets = {
        "Separate sensor topology": train_features,
        "Euclidean joint topology": joint_features,
        "Max-coupled topology": coupled_features,
        "Desynchronized max-coupled topology": coupled_primary_desynchronized,
        "Separate + max-coupled topology": np.column_stack(
            (train_features, coupled_features)
        ),
        "Separate + desynchronized max-coupled topology": np.column_stack(
            (train_features, coupled_primary_desynchronized)
        ),
    }
    coupled_predictions = {
        name: fixed_grouped_predictions(
            features,
            train_labels,
            train_subjects,
            C=parameters["svc__C"],
            gamma=parameters["svc__gamma"],
        )
        for name, features in coupled_feature_sets.items()
    }
    coupled_scores = {
        name: classification_metrics(train_labels, predictions)
        for name, predictions in coupled_predictions.items()
    }
    coupled_weight_sensitivity = {
        f"acceleration weight {weight:.2f}": fixed_grouped_cv(
            features,
            train_labels,
            train_subjects,
            C=parameters["svc__C"],
            gamma=parameters["svc__gamma"],
        )
        for weight, features in coupled_by_weight.items()
    }
    coupled_desynchronization_sensitivity = {}
    for seed, features in coupled_desynchronized_by_seed.items():
        if seed == SEED:
            desynchronized_predictions = coupled_predictions[
                "Desynchronized max-coupled topology"
            ]
            combined_predictions = coupled_predictions[
                "Separate + desynchronized max-coupled topology"
            ]
        else:
            desynchronized_predictions = fixed_grouped_predictions(
                features,
                train_labels,
                train_subjects,
                C=parameters["svc__C"],
                gamma=parameters["svc__gamma"],
            )
            combined_predictions = fixed_grouped_predictions(
                np.column_stack((train_features, features)),
                train_labels,
                train_subjects,
                C=parameters["svc__C"],
                gamma=parameters["svc__gamma"],
            )
        coupled_desynchronization_sensitivity[str(seed)] = {
            "coupled": classification_metrics(train_labels, desynchronized_predictions),
            "separate_plus_coupled": classification_metrics(
                train_labels, combined_predictions
            ),
        }

    coupled_minus_desynchronized = paired_subject_bootstrap_difference(
        coupled_predictions["Max-coupled topology"],
        coupled_predictions["Desynchronized max-coupled topology"],
        train_labels,
        train_subjects,
    )
    coupled_combined_minus_separate = paired_subject_bootstrap_difference(
        coupled_predictions["Separate + max-coupled topology"],
        coupled_predictions["Separate sensor topology"],
        train_labels,
        train_subjects,
    )
    coupled_combined_synchronized_minus_desynchronized = (
        paired_subject_bootstrap_difference(
            coupled_predictions["Separate + max-coupled topology"],
            coupled_predictions["Separate + desynchronized max-coupled topology"],
            train_labels,
            train_subjects,
        )
    )

    print("Stable-rank nearest-prototype classifier", flush=True)
    classifier = prototype_classifier(
        train_features,
        train_labels,
        train_subjects,
        test_features,
        test_labels,
        test_subjects,
    )

    prototypes, prototype_labels, prototype_subjects = subject_activity_prototypes(
        train_features, train_labels, train_subjects
    )
    distance_matrix = activity_distance_matrix(prototypes, prototype_labels)

    print("Subject-wise label-permutation null", flush=True)
    permutation = prototype_permutation_test(
        prototypes, prototype_labels, prototype_subjects
    )

    output = {
        "ablation_protocol": "Grouped development CV with the primary frozen SVM settings.",
        "ablations": ablations,
        "l1_nearest_prototype": classifier,
        "prototype_activity_distance_matrix": distance_matrix.tolist(),
        "prototype_permutation_test": permutation,
        "joint_representation": {
            "protocol": (
                "Post-hoc grouped development CV with primary frozen SVM settings; "
                "the official test subjects are not used."
            ),
            "configuration": {
                "embedding_dimension": 3,
                "lag": 15,
                "equal_weight_acceleration": 0.5,
                "joint_cloud_shape": [98, 18],
                "joint_feature_width": 128,
                "desynchronization_primary_seed": SEED,
                "desynchronization_sensitivity_seeds": list(
                    desynchronization_seeds
                ),
            },
            "scores": joint_scores,
            "weight_sensitivity": joint_weight_sensitivity,
            "combined_minus_separate": combined_minus_separate,
            "desynchronization_control": {
                "construction": (
                    "Derange the 98 gyroscope delay points within each window "
                    "after separate block normalization. Both marginal point "
                    "clouds are unchanged; only cross-sensor pairing is removed."
                ),
                "seed_scores": desynchronization_sensitivity,
                "synchronized_minus_desynchronized_joint": (
                    synchronized_minus_desynchronized
                ),
                "combined_synchronized_minus_desynchronized": (
                    combined_synchronized_minus_desynchronized
                ),
                "desynchronized_combined_minus_separate": (
                    desynchronized_combined_minus_separate
                ),
            },
        },
        "coupled_max_representation": {
            "protocol": (
                "Post-hoc grouped development CV with primary frozen SVM settings; "
                "the official test subjects are not used."
            ),
            "construction": (
                "For paired normalized delay clouds, use the maximum of the "
                "weighted acceleration and gyroscope distances. At equal weight, "
                "an edge appears exactly when both marginal distances are at most "
                "the filtration threshold."
            ),
            "configuration": {
                "embedding_dimension": 3,
                "lag": 15,
                "equal_weight_acceleration": 0.5,
                "distance_matrix_shape": [98, 98],
                "coupled_feature_width": 128,
                "desynchronization_primary_seed": SEED,
                "desynchronization_sensitivity_seeds": list(
                    desynchronization_seeds
                ),
            },
            "scores": coupled_scores,
            "weight_sensitivity": coupled_weight_sensitivity,
            "coupled_minus_desynchronized": coupled_minus_desynchronized,
            "combined_minus_separate": coupled_combined_minus_separate,
            "combined_synchronized_minus_desynchronized": (
                coupled_combined_synchronized_minus_desynchronized
            ),
            "desynchronization_seed_scores": (
                coupled_desynchronization_sensitivity
            ),
        },
    }
    result_path.write_text(json.dumps(output, indent=2))

    print("Extension analysis complete.", flush=True)
    return output
