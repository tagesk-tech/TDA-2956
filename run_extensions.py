"""Post-hoc stable-rank diagnostics using the cached primary features."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from data_loader import load_inertial_data
from evaluation import fixed_grouped_cv, fixed_grouped_predictions
from features import (
    STABLE_RANK_GRID,
    coupled_sensor_stable_ranks,
    joint_sensor_stable_ranks,
    stable_rank_blocks,
    stable_rank_l1,
)


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
SEED = 2956


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


def subject_activity_prototypes(features, labels, subjects):
    """Average windows so every row represents one subject performing one activity."""
    rows, activities, groups = [], [], []
    for subject in np.unique(subjects):
        for activity in range(1, 7):
            mask = (subjects == subject) & (labels == activity)
            rows.append(features[mask].mean(axis=0))
            activities.append(activity)
            groups.append(subject)
    return np.asarray(rows), np.asarray(activities), np.asarray(groups)


def nearest_prototype(train_features, train_labels, query_features, chunk_size=512):
    """Classify by stable-rank distance on [0, 2] to six class means."""
    class_means = np.stack(
        [train_features[train_labels == activity].mean(axis=0) for activity in range(1, 7)]
    )
    predictions = []
    for start in range(0, len(query_features), chunk_size):
        query = query_features[start : start + chunk_size]
        distances = stable_rank_l1(query[:, None, :], class_means[None, :, :])
        predictions.append(np.argmin(distances, axis=1) + 1)
    return np.concatenate(predictions)


def _metrics(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def _subject_bootstrap(labels, predictions, subjects, repeats=2000):
    rng = np.random.default_rng(SEED)
    subject_ids = np.unique(subjects)
    scores = []
    for _ in range(repeats):
        sampled = rng.choice(subject_ids, len(subject_ids), replace=True)
        indices = np.concatenate([np.flatnonzero(subjects == subject) for subject in sampled])
        scores.append(balanced_accuracy_score(labels[indices], predictions[indices]))
    return np.quantile(scores, (0.025, 0.975)).tolist()


def paired_subject_bootstrap_difference(
    first_predictions,
    second_predictions,
    labels,
    subjects,
    repeats=10000,
):
    """Bootstrap the mean subject-level balanced-accuracy difference."""
    differences = []
    for subject in np.unique(subjects):
        mask = subjects == subject
        first_score = balanced_accuracy_score(labels[mask], first_predictions[mask])
        second_score = balanced_accuracy_score(labels[mask], second_predictions[mask])
        differences.append(first_score - second_score)

    differences = np.asarray(differences)
    rng = np.random.default_rng(SEED)
    sampled = rng.choice(
        differences,
        size=(repeats, len(differences)),
        replace=True,
    ).mean(axis=1)
    return {
        "mean": float(differences.mean()),
        "ci": np.quantile(sampled, (0.025, 0.975)).tolist(),
        "subject_differences": differences.tolist(),
    }


def prototype_classifier(train_features, train_labels, train_subjects, test_features, test_labels, test_subjects):
    """Subject-grouped CV plus one official-test nearest-prototype evaluation."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_predictions = np.empty_like(train_labels)
    for fit, validation in splitter.split(train_features, train_labels, train_subjects):
        fit_prototypes, fit_activities, _ = subject_activity_prototypes(
            train_features[fit], train_labels[fit], train_subjects[fit]
        )
        cv_predictions[validation] = nearest_prototype(
            fit_prototypes, fit_activities, train_features[validation]
        )

    prototypes, activities, _ = subject_activity_prototypes(
        train_features, train_labels, train_subjects
    )
    test_predictions = nearest_prototype(prototypes, activities, test_features)
    return {
        "grouped_cv": _metrics(train_labels, cv_predictions),
        "official_test": _metrics(test_labels, test_predictions),
        "test_balanced_accuracy_ci": _subject_bootstrap(
            test_labels, test_predictions, test_subjects
        ),
        "test_confusion_matrix": confusion_matrix(
            test_labels, test_predictions, labels=np.arange(1, 7), normalize="true"
        ).tolist(),
    }


def prototype_permutation_test(features, labels, subjects, repeats=2000):
    """Test cross-subject activity structure after independently relabeling each subject."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(features, labels, subjects))

    def score(current_labels):
        predictions = np.empty_like(current_labels)
        for fit, validation in splits:
            predictions[validation] = nearest_prototype(
                features[fit], current_labels[fit], features[validation]
            )
        return balanced_accuracy_score(current_labels, predictions)

    observed = score(labels)
    rng = np.random.default_rng(SEED)
    null_scores = np.empty(repeats)
    for repeat in range(repeats):
        permuted = labels.copy()
        for subject in np.unique(subjects):
            mask = subjects == subject
            mapping = rng.permutation(np.arange(1, 7))
            permuted[mask] = mapping[labels[mask] - 1]
        null_scores[repeat] = score(permuted)

    return {
        "observed_balanced_accuracy": float(observed),
        "null_mean": float(null_scores.mean()),
        "null_95_interval": np.quantile(null_scores, (0.025, 0.975)).tolist(),
        "p_value": float((1 + np.sum(null_scores >= observed)) / (repeats + 1)),
        "repeats": repeats,
        "null_scores": null_scores.tolist(),
    }


def activity_distance_matrix(prototypes, activities):
    """Return stable-rank distances on [0, 2] between activity signatures."""
    activity_means = np.stack(
        [prototypes[activities == activity].mean(axis=0) for activity in range(1, 7)]
    )
    return stable_rank_l1(activity_means[:, None, :], activity_means[None, :, :])


def main():
    train_signals, train_labels, train_subjects = load_inertial_data("train")
    test_signals, test_labels, test_subjects = load_inertial_data("test")
    del test_signals

    cached = np.load(ARTIFACTS / "delay_normalized_m3_lag15.npz")
    train_features = cached["train_curves"]
    test_features = cached["test_curves"]

    print("Homology and sensor ablations", flush=True)
    primary = json.loads((RESULTS / "primary_results.json").read_text())
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
        name: _metrics(train_labels, predictions)
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
            "joint": _metrics(train_labels, joint_desynchronized_predictions),
            "separate_plus_joint": _metrics(
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
        name: _metrics(train_labels, predictions)
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
            "coupled": _metrics(train_labels, desynchronized_predictions),
            "separate_plus_coupled": _metrics(train_labels, combined_predictions),
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
    (RESULTS / "extension_results.json").write_text(json.dumps(output, indent=2))

    print("Extension analysis complete.", flush=True)


if __name__ == "__main__":
    main()
