"""Subject-wise model selection and evaluation."""

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


def _metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


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
