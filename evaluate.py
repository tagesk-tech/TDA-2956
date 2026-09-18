"""One fixed subject-grouped evaluation protocol."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import ACTIVITIES
from tda import stable_rank_distance


SEED = 2956


def model():
    return make_pipeline(
        StandardScaler(),
        SVC(C=10.0, gamma=0.001, kernel="rbf", class_weight="balanced"),
    )


def metrics(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def evaluate_development_feature_set(features, labels, subjects):
    """Five-fold subject-grouped development predictions and metrics."""
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    predictions = cross_val_predict(
        model(), features, labels, groups=subjects, cv=folds, n_jobs=1,
    )
    return metrics(labels, predictions)


def evaluate_feature_set(
    train_features,
    train_labels,
    train_subjects,
    test_features,
    test_labels,
):
    """Grouped development predictions and one official test evaluation."""
    development = evaluate_development_feature_set(
        train_features, train_labels, train_subjects
    )
    fitted = model().fit(train_features, train_labels)
    test_predictions = fitted.predict(test_features)
    return {
        "development": development,
        "official_test": metrics(test_labels, test_predictions),
        "test_confusion_matrix": confusion_matrix(
            test_labels, test_predictions, labels=sorted(ACTIVITIES), normalize="true"
        ).tolist(),
    }


def subject_activity_prototypes(features, labels, subjects):
    rows, row_labels = [], []
    for subject in np.unique(subjects):
        for activity in ACTIVITIES:
            mask = (subjects == subject) & (labels == activity)
            rows.append(features[mask].mean(axis=0))
            row_labels.append(activity)
    return np.asarray(rows), np.asarray(row_labels)


def activity_distance_matrix(features, labels, subjects):
    """Mean cross-subject stable-rank distance for every activity pair."""
    prototypes, prototype_labels = subject_activity_prototypes(
        features, labels, subjects
    )
    matrix = np.empty((6, 6))
    for first in ACTIVITIES:
        left = prototypes[prototype_labels == first]
        for second in ACTIVITIES:
            right = prototypes[prototype_labels == second]
            distances = stable_rank_distance(left[:, None, :], right[None, :, :])
            if first == second:
                distances = distances[~np.eye(len(left), dtype=bool)]
            matrix[first - 1, second - 1] = distances.mean()
    return matrix
