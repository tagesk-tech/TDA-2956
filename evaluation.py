"""Reusable subject-wise model selection and statistical comparisons."""

from itertools import product

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import stable_rank_l1


def _metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def classification_metrics(y_true, y_pred):
    """Return the three classification scores reported by the project."""
    return _metrics(y_true, y_pred)


def _subject_bootstrap(y_true, y_pred, subjects, repeats=2000, seed=2956):
    rng = np.random.default_rng(seed)
    subject_ids = np.unique(subjects)
    samples = []

    for _ in range(repeats):
        drawn = rng.choice(subject_ids, size=len(subject_ids), replace=True)
        indices = np.concatenate([np.flatnonzero(subjects == subject) for subject in drawn])
        samples.append(balanced_accuracy_score(y_true[indices], y_pred[indices]))

    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


SEED = 2956
PARAMETER_GRID = {
    "svc__C": (0.1, 1.0, 10.0),
    "svc__gamma": ("scale", 0.001, 0.01),
}


def _model_search(folds, seed):
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", cache_size=2000),
    )
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return GridSearchCV(
        pipeline,
        PARAMETER_GRID,
        scoring="balanced_accuracy",
        cv=splitter,
        n_jobs=-1,
        refit=True,
    )


def _nested_predictions(features, labels, subjects):
    """Predict each training subject with parameters chosen without that subject."""
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    predictions = np.empty_like(labels)
    fold_parameters = []

    for fold, (fit, validation) in enumerate(outer.split(features, labels, subjects)):
        search = _model_search(folds=4, seed=SEED + fold + 1)
        search.fit(features[fit], labels[fit], groups=subjects[fit])
        predictions[validation] = search.predict(features[validation])
        fold_parameters.append(search.best_params_)

    return predictions, fold_parameters


def evaluate_feature_set(X_train, y_train, train_subjects, X_test, y_test, test_subjects):
    """Run nested grouped CV, then fit once and score the official test subjects."""
    cv_predictions, fold_parameters = _nested_predictions(
        X_train, y_train, train_subjects
    )

    final_search = _model_search(folds=5, seed=SEED)
    final_search.fit(X_train, y_train, groups=train_subjects)
    selected_model = final_search.best_estimator_
    test_predictions = selected_model.predict(X_test)

    result = {
        "parameters": final_search.best_params_,
        "selection_score": float(final_search.best_score_),
        "outer_fold_parameters": fold_parameters,
        "grouped_cv": _metrics(y_train, cv_predictions),
        "official_test": _metrics(y_test, test_predictions),
        "test_balanced_accuracy_ci": _subject_bootstrap(
            y_test, test_predictions, test_subjects
        ),
        "test_confusion_matrix": confusion_matrix(
            y_test, test_predictions, labels=np.arange(1, 7), normalize="true"
        ).tolist(),
        "test_predictions": test_predictions.tolist(),
    }
    return result, selected_model


def fixed_grouped_predictions(X, labels, subjects, C=10.0, gamma=0.001):
    """Return frozen-SVM predictions under the primary subject-grouped folds."""
    grouped_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced", cache_size=2000),
    )
    return cross_val_predict(
        model, X, labels, groups=subjects, cv=grouped_cv, n_jobs=-1
    )


def fixed_grouped_cv(X, labels, subjects, C=10.0, gamma=0.001):
    """Evaluate a frozen SVM under the primary subject-grouped folds."""
    predictions = fixed_grouped_predictions(X, labels, subjects, C, gamma)
    return _metrics(labels, predictions)


def subject_activity_prototypes(features, labels, subjects):
    """Average windows so each row represents one subject and activity."""
    rows, activities, groups = [], [], []
    for subject in np.unique(subjects):
        for activity in range(1, 7):
            mask = (subjects == subject) & (labels == activity)
            rows.append(features[mask].mean(axis=0))
            activities.append(activity)
            groups.append(subject)
    return np.asarray(rows), np.asarray(activities), np.asarray(groups)


def nearest_prototype(train_features, train_labels, query_features, chunk_size=512):
    """Classify by stable-rank distance to six activity means."""
    class_means = np.stack([
        train_features[train_labels == activity].mean(axis=0)
        for activity in range(1, 7)
    ])
    predictions = []
    for start in range(0, len(query_features), chunk_size):
        query = query_features[start : start + chunk_size]
        distances = stable_rank_l1(query[:, None, :], class_means[None, :, :])
        predictions.append(np.argmin(distances, axis=1) + 1)
    return np.concatenate(predictions)


def paired_subject_bootstrap_difference(
    first_predictions,
    second_predictions,
    labels,
    subjects,
    repeats=10000,
    seed=SEED,
):
    """Bootstrap the mean subject-level balanced-accuracy difference."""
    differences = []
    for subject in np.unique(subjects):
        mask = subjects == subject
        first = balanced_accuracy_score(labels[mask], first_predictions[mask])
        second = balanced_accuracy_score(labels[mask], second_predictions[mask])
        differences.append(first - second)

    differences = np.asarray(differences)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        differences, size=(repeats, len(differences)), replace=True
    ).mean(axis=1)
    return {
        "mean": float(differences.mean()),
        "ci": np.quantile(sampled, (0.025, 0.975)).tolist(),
        "subject_differences": differences.tolist(),
    }


def paired_subject_comparison(
    first_predictions,
    second_predictions,
    labels,
    subjects,
    repeats=10000,
    seed=SEED,
):
    """Add an exact paired sign-flip test to the subject bootstrap."""
    result = paired_subject_bootstrap_difference(
        first_predictions,
        second_predictions,
        labels,
        subjects,
        repeats,
        seed,
    )
    differences = np.asarray(result["subject_differences"])
    observed = abs(differences.mean())
    signed_means = np.asarray([
        np.mean(differences * signs)
        for signs in product((-1.0, 1.0), repeat=len(differences))
    ])
    result["exact_two_sided_sign_flip_p"] = float(
        np.mean(np.abs(signed_means) >= observed - 1e-15)
    )
    return result


def prototype_classifier(
    train_features,
    train_labels,
    train_subjects,
    test_features,
    test_labels,
    test_subjects,
):
    """Evaluate a nearest-prototype rule with subject-grouped validation."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_predictions = np.empty_like(train_labels)
    for fit, validation in splitter.split(train_features, train_labels, train_subjects):
        prototypes, activities, _ = subject_activity_prototypes(
            train_features[fit], train_labels[fit], train_subjects[fit]
        )
        cv_predictions[validation] = nearest_prototype(
            prototypes, activities, train_features[validation]
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


def prototype_permutation_test(features, labels, subjects, repeats=2000, seed=SEED):
    """Test cross-subject structure after independently relabeling each subject."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(splitter.split(features, labels, subjects))

    def score(current_labels):
        predictions = np.empty_like(current_labels)
        for fit, validation in splits:
            predictions[validation] = nearest_prototype(
                features[fit], current_labels[fit], features[validation]
            )
        return balanced_accuracy_score(current_labels, predictions)

    observed = score(labels)
    rng = np.random.default_rng(seed)
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
    """Stable-rank distances between six subject-balanced activity signatures."""
    means = np.stack([
        prototypes[activities == activity].mean(axis=0)
        for activity in range(1, 7)
    ])
    return stable_rank_l1(means[:, None, :], means[None, :, :])


def cross_subject_activity_distances(features, labels, subjects):
    """Compare subject-activity signatures without same-subject pairs."""
    subject_ids = np.unique(subjects)
    prototypes = {
        (subject, activity): features[
            (subjects == subject) & (labels == activity)
        ].mean(axis=0)
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


def activity_effect(subject_activity_values):
    """Partial eta squared for activity after removing each subject mean."""
    values = np.asarray(subject_activity_values)
    grand_mean = values.mean(axis=(0, 1))
    activity_means = values.mean(axis=0)
    subject_means = values.mean(axis=1)
    residual = (
        values
        - subject_means[:, None, :]
        - activity_means[None, :, :]
        + grand_mean[None, None, :]
    )
    activity_ss = len(values) * np.sum((activity_means - grand_mean) ** 2, axis=0)
    error_ss = np.sum(residual**2, axis=(0, 1))
    return activity_ss / (activity_ss + error_ss)
