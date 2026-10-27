import argparse
import json
from pathlib import Path

import numpy as np

from data import load_engineered_features, load_split
from evaluate import development_balanced_accuracy, evaluate
from tda import (
    GRID,
    feature_matrices,
    first_nonpositive_lag,
    joint_feature_matrices,
    mean_autocorrelation,
)


SEED = 2956
WEIGHTS = np.linspace(0.0, 1.0, 11)
OUTPUT = Path(__file__).parent / "output"


def cache(path, compute, recompute):
    if path.exists() and not recompute:
        print(f"load {path.name}", flush=True)
        with np.load(path, allow_pickle=False) as saved:
            return {name: saved[name] for name in saved.files}
    print(f"compute {path.name}", flush=True)
    values = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)
    return values


def sensor_features(name, train, test, recompute, keep_scales=False, **options):
    def compute():
        train_ranks, train_scales = feature_matrices(
            train, progress=True, **options
        )
        test_ranks, test_scales = feature_matrices(
            test, progress=True, **options
        )
        values = {
            "train_ranks": train_ranks,
            "test_ranks": test_ranks,
        }
        if keep_scales:
            values["train_scales"] = train_scales
            values["test_scales"] = test_scales
        return values

    return cache(OUTPUT / "cache" / f"{name}.npz", compute, recompute)


def search_weights(name, weights, joint_ranks, labels, subjects, base=None):
    scores = []
    for weight, ranks in zip(weights, joint_ranks):
        print(f"{name} {weight:.1f}", flush=True)
        features = ranks if base is None else np.column_stack((base, ranks))
        scores.append({
            "acceleration_weight": float(weight),
            "balanced_accuracy": development_balanced_accuracy(
                features, labels, subjects
            ),
        })
    selected = max(range(len(scores)), key=lambda index: (
        scores[index]["balanced_accuracy"],
        -abs(scores[index]["acceleration_weight"] - 0.5),
    ))
    return scores, selected


def run(recompute=False):
    train, train_labels, train_subjects = load_split("train")
    test, test_labels, test_subjects = load_split("test")

    autocorrelation = mean_autocorrelation(train["body_acc"])
    lag = first_nonpositive_lag(autocorrelation)

    delay = sensor_features(
        "delay_m3_lag15_f47",
        train,
        test,
        recompute,
        keep_scales=True,
        m=3,
        lag=15,
    )
    instantaneous = sensor_features(
        "instantaneous_m1_lag1_f47",
        train,
        test,
        recompute,
        m=1,
        lag=1,
    )
    shuffled = sensor_features(
        "shuffled_m3_lag15_seed2956_f47",
        train,
        test,
        recompute,
        m=3,
        lag=15,
        shuffle_seed=SEED,
    )

    def compute_joint_train():
        middle = joint_feature_matrices(
            train, WEIGHTS[1:-1], m=3, lag=15, progress=True
        )
        ranks = np.empty((len(WEIGHTS), len(train_labels), 2 * len(GRID)))
        ranks[0] = delay["train_ranks"][:, 2 * len(GRID):]
        ranks[-1] = delay["train_ranks"][:, :2 * len(GRID)]
        ranks[1:-1] = middle
        return {"weights": WEIGHTS, "train_ranks": ranks}

    joint_train = cache(
        OUTPUT / "cache" / "joint_train_m3_lag15_weights_step0p1_f47.npz",
        compute_joint_train,
        recompute,
    )
    joint_search, joint_index = search_weights(
        "joint",
        joint_train["weights"],
        joint_train["train_ranks"],
        train_labels,
        train_subjects,
    )
    hybrid_search, hybrid_index = search_weights(
        "hybrid",
        joint_train["weights"],
        joint_train["train_ranks"],
        train_labels,
        train_subjects,
        delay["train_ranks"],
    )
    joint_weight = joint_search[joint_index]["acceleration_weight"]
    hybrid_weight = hybrid_search[hybrid_index]["acceleration_weight"]

    def joint_test(weight):
        label = f"{weight:.1f}".replace(".", "p")
        path = OUTPUT / "cache" / (
            f"joint_test_m3_lag15_accweight{label}_f47.npz"
        )
        return cache(
            path,
            lambda: {
                "test_ranks": joint_feature_matrices(
                    test, [weight], m=3, lag=15, progress=True
                )[0]
            },
            recompute,
        )

    joint_test_ranks = joint_test(joint_weight)["test_ranks"]
    hybrid_test_ranks = joint_test(hybrid_weight)["test_ranks"]
    uci_train = load_engineered_features("train")
    uci_test = load_engineered_features("test")

    representations = {
        "Instantaneous TDA": (
            instantaneous["train_ranks"], instantaneous["test_ranks"]
        ),
        "Delay TDA": (delay["train_ranks"], delay["test_ranks"]),
        "Joint-sensor TDA": (
            joint_train["train_ranks"][joint_index], joint_test_ranks
        ),
        "Hybrid TDA": (
            np.column_stack((
                delay["train_ranks"], joint_train["train_ranks"][hybrid_index]
            )),
            np.column_stack((delay["test_ranks"], hybrid_test_ranks)),
        ),
        "Hybrid TDA + scale": (
            np.column_stack((
                delay["train_ranks"],
                joint_train["train_ranks"][hybrid_index],
                delay["train_scales"],
            )),
            np.column_stack((
                delay["test_ranks"],
                hybrid_test_ranks,
                delay["test_scales"],
            )),
        ),
        "Shuffled-delay TDA": (
            shuffled["train_ranks"], shuffled["test_ranks"]
        ),
        "Delay TDA + scale": (
            np.column_stack((delay["train_ranks"], delay["train_scales"])),
            np.column_stack((delay["test_ranks"], delay["test_scales"])),
        ),
        "UCI 561 baseline": (uci_train, uci_test),
    }
    models = {}
    for name, (train_features, test_features) in representations.items():
        print(name, flush=True)
        models[name] = evaluate(
            train_features,
            train_labels,
            train_subjects,
            test_features,
            test_labels,
        )

    results = {
        "configuration": {
            "seed": SEED,
            "embedding_dimension": 3,
            "lag": 15,
            "measured_first_nonpositive_lag": lag,
            "stable_rank_grid": GRID.tolist(),
            "train_subjects": np.unique(train_subjects).tolist(),
            "test_subjects": np.unique(test_subjects).tolist(),
        },
        "models": models,
        "joint_sensor_weight_search": {
            "selected_acceleration_weight": joint_weight,
            "selected_gyroscope_weight": 1.0 - joint_weight,
            "grid": joint_search,
        },
        "hybrid_sensor_weight_search": {
            "selected_acceleration_weight": hybrid_weight,
            "selected_gyroscope_weight": 1.0 - hybrid_weight,
            "grid": hybrid_search,
        },
    }
    result_path = OUTPUT / "results.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n")
    print(result_path)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    run(parser.parse_args().recompute)


if __name__ == "__main__":
    main()
