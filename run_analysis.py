"""Run the analysis specified in ANALYSIS_SPEC.md without a notebook."""

import argparse
import json
from pathlib import Path

import numpy as np

from data import (
    first_subject_activity_windows,
    load_engineered_features,
    load_split,
)
from evaluate import (
    activity_distance_matrix,
    evaluate_development_feature_set,
    evaluate_feature_set,
)
from tda import (
    GRID,
    feature_matrices,
    first_nonpositive_lag,
    mean_autocorrelation,
    joint_feature_matrices,
    stable_rank_distance,
)


SEED = 2956
JOINT_WEIGHTS = np.linspace(0.0, 1.0, 11)
OUTPUT = Path(__file__).parent / "output"


def _cache(path, compute, recompute):
    if path.exists() and not recompute:
        print(f"Loading {path.name}", flush=True)
        with np.load(path, allow_pickle=False) as saved:
            return {name: saved[name] for name in saved.files}
    print(f"Computing {path.name}", flush=True)
    values = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)
    return values


def _extract_pair(name, train, test, recompute, **options):
    def compute():
        train_values = feature_matrices(train, progress=True, **options)
        test_values = feature_matrices(test, progress=True, **options)
        result = {}
        for split, values in (("train", train_values), ("test", test_values)):
            ranks, scales, dynamics = values
            result[f"{split}_ranks"] = ranks
            result[f"{split}_scales"] = scales
            if dynamics is not None:
                result[f"{split}_dynamics"] = dynamics
        return result
    return _cache(OUTPUT / "cache" / f"{name}.npz", compute, recompute)


def _finite_arrays(values):
    return all(np.isfinite(array).all() for array in values.values())


def _search_joint_weights(name, weights, joint_ranks, labels, subjects, base=None):
    scores = []
    for weight, ranks in zip(weights, joint_ranks):
        print(f"Evaluating {name} acceleration weight {weight:.1f}", flush=True)
        features = ranks if base is None else np.column_stack((base, ranks))
        scores.append({
            "acceleration_weight": float(weight),
            "gyroscope_weight": float(1.0 - weight),
            **evaluate_development_feature_set(features, labels, subjects),
        })
    index = max(range(len(scores)), key=lambda i: (
        scores[i]["balanced_accuracy"],
        -abs(scores[i]["acceleration_weight"] - 0.5),
    ))
    return scores, index


def run(smoke=False, recompute=False):
    train, train_labels, train_subjects = load_split("train")
    test, test_labels, test_subjects = load_split("test")
    if not set(train_subjects).isdisjoint(test_subjects):
        raise RuntimeError("official train and test subjects overlap")

    prefix = "smoke_" if smoke else ""
    if smoke:
        train_index = first_subject_activity_windows(train_labels, train_subjects)
        test_index = first_subject_activity_windows(test_labels, test_subjects)
        train = {name: values[train_index] for name, values in train.items()}
        test = {name: values[test_index] for name, values in test.items()}
        train_labels, train_subjects = train_labels[train_index], train_subjects[train_index]
        test_labels, test_subjects = test_labels[test_index], test_subjects[test_index]

    autocorrelation = mean_autocorrelation(train["body_acc"])
    chosen_lag = first_nonpositive_lag(autocorrelation)
    if not smoke and chosen_lag != 15:
        raise RuntimeError(f"expected development-data lag 15, obtained {chosen_lag}")

    delay = _extract_pair(
        prefix + "delay_m3_lag15_f47", train, test, recompute,
        m=3, lag=15, include_dynamics=True,
    )
    instantaneous = _extract_pair(
        prefix + "instantaneous_m1_lag1_f47", train, test, recompute,
        m=1, lag=1,
    )
    shuffled = _extract_pair(
        prefix + "shuffled_m3_lag15_seed2956_f47", train, test, recompute,
        m=3, lag=15, shuffle_seed=SEED,
    )

    def compute_joint_train():
        middle_ranks, scales = joint_feature_matrices(
            train, JOINT_WEIGHTS[1:-1], m=3, lag=15, progress=True
        )
        all_ranks = np.empty((len(JOINT_WEIGHTS), len(train_labels), 2 * len(GRID)))
        all_ranks[0] = delay["train_ranks"][:, 2 * len(GRID):]
        all_ranks[-1] = delay["train_ranks"][:, :2 * len(GRID)]
        all_ranks[1:-1] = middle_ranks
        return {"weights": JOINT_WEIGHTS, "train_ranks": all_ranks, "train_scales": scales}

    joint_train = _cache(
        OUTPUT / "cache" / f"{prefix}joint_train_m3_lag15_weights_step0p1_f47.npz",
        compute_joint_train,
        recompute,
    )
    for values in (delay, instantaneous, shuffled, joint_train):
        if not _finite_arrays(values):
            raise RuntimeError("feature extraction produced a non-finite value")

    weight_search, selected_index = _search_joint_weights(
        "joint", joint_train["weights"], joint_train["train_ranks"],
        train_labels, train_subjects,
    )
    hybrid_search, hybrid_index = _search_joint_weights(
        "hybrid", joint_train["weights"], joint_train["train_ranks"],
        train_labels, train_subjects, base=delay["train_ranks"],
    )
    selected_weight = weight_search[selected_index]["acceleration_weight"]
    hybrid_weight = hybrid_search[hybrid_index]["acceleration_weight"]
    print(f"Selected acceleration weights: joint={selected_weight:.1f}, hybrid={hybrid_weight:.1f}", flush=True)

    def joint_test(weight):
        def compute():
            ranks, scales = joint_feature_matrices(
                test, [weight], m=3, lag=15, progress=True
            )
            return {"test_ranks": ranks[0], "test_scales": scales}
        label = f"{weight:.1f}".replace(".", "p")
        path = OUTPUT / "cache" / f"{prefix}joint_test_m3_lag15_accweight{label}_f47.npz"
        return _cache(path, compute, recompute)

    joint_test_values = joint_test(selected_weight)
    hybrid_test_values = joint_test(hybrid_weight)

    if smoke:
        uci_train = load_engineered_features("train")[train_index]
        uci_test = load_engineered_features("test")[test_index]
    else:
        uci_train = load_engineered_features("train")
        uci_test = load_engineered_features("test")

    representations = {
        "Instantaneous TDA": (
            instantaneous["train_ranks"], instantaneous["test_ranks"]),
        "Delay TDA": (delay["train_ranks"], delay["test_ranks"]),
        "Joint-sensor TDA": (
            joint_train["train_ranks"][selected_index],
            joint_test_values["test_ranks"],
        ),
        "Hybrid TDA": (
            np.column_stack((delay["train_ranks"], joint_train["train_ranks"][hybrid_index])),
            np.column_stack((delay["test_ranks"], hybrid_test_values["test_ranks"])),
        ),
        "Hybrid TDA + scale": (
            np.column_stack((delay["train_ranks"], joint_train["train_ranks"][hybrid_index], delay["train_scales"])),
            np.column_stack((delay["test_ranks"], hybrid_test_values["test_ranks"], delay["test_scales"])),
        ),
        "Shuffled-delay TDA": (
            shuffled["train_ranks"], shuffled["test_ranks"]),
        "Delay TDA + scale": (
            np.column_stack((delay["train_ranks"], delay["train_scales"])),
            np.column_stack((delay["test_ranks"], delay["test_scales"]))),
        "Topological dynamics": (
            delay["train_dynamics"], delay["test_dynamics"]),
        "UCI 561 baseline": (uci_train, uci_test),
    }
    scores = {}
    for name, (train_features, test_features) in representations.items():
        print(f"Evaluating {name}", flush=True)
        scores[name] = evaluate_feature_set(
            train_features, train_labels, train_subjects,
            test_features, test_labels,
        )

    distances = activity_distance_matrix(
        delay["train_ranks"], train_labels, train_subjects
    )

    rng = np.random.default_rng(SEED)
    noise_index = np.concatenate([
        rng.choice(np.flatnonzero(train_labels == activity),
                   min(10, np.sum(train_labels == activity)), replace=False)
        for activity in range(1, 7)
    ])
    noisy = {}
    for sensor, windows in train.items():
        selected = windows[noise_index]
        noise = rng.normal(size=selected.shape)
        noisy[sensor] = selected + 0.05 * selected.std(
            axis=1, keepdims=True
        ) * noise
    noisy_ranks, _, _ = feature_matrices(noisy, m=3, lag=15)
    displacement = stable_rank_distance(
        noisy_ranks, delay["train_ranks"][noise_index]
    )

    results = {
        "mode": "smoke" if smoke else "full",
        "configuration": {
            "seed": SEED,
            "embedding_dimension": 3,
            "lag": 15,
            "measured_first_nonpositive_lag": chosen_lag,
            "stable_rank_grid": GRID.tolist(),
            "train_subjects": np.unique(train_subjects).tolist(),
            "test_subjects": np.unique(test_subjects).tolist(),
        },
        "models": scores,
        "joint_sensor_weight_search": {
            "selection_data": "subject-grouped development folds only",
            "selected_acceleration_weight": selected_weight,
            "selected_gyroscope_weight": 1.0 - selected_weight,
            "grid": weight_search,
        },
        "hybrid_sensor_weight_search": {
            "selection_data": "subject-grouped development folds only",
            "selected_acceleration_weight": hybrid_weight,
            "selected_gyroscope_weight": 1.0 - hybrid_weight,
            "grid": hybrid_search,
        },
        "activity_distance_matrix": distances.tolist(),
        "noise_5_percent": {
            "sample_size": int(len(displacement)),
            "mean_distance": float(displacement.mean()),
            "q025": float(np.quantile(displacement, 0.025)),
            "q975": float(np.quantile(displacement, 0.975)),
        },
    }
    OUTPUT.mkdir(exist_ok=True)
    result_path = OUTPUT / ("smoke_results.json" if smoke else "results.json")
    result_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {result_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    arguments = parser.parse_args()
    run(smoke=arguments.smoke, recompute=arguments.recompute)


if __name__ == "__main__":
    main()
