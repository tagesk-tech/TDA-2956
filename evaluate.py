from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 2956


def model():
    return make_pipeline(
        StandardScaler(),
        SVC(C=10.0, gamma=0.001, kernel="rbf", class_weight="balanced"),
    )


def development_balanced_accuracy(features, labels, subjects):
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    predictions = cross_val_predict(
        model(), features, labels, groups=subjects, cv=folds, n_jobs=1,
    )
    return float(balanced_accuracy_score(labels, predictions))


def evaluate(
    train_features,
    train_labels,
    train_subjects,
    test_features,
    test_labels,
):
    development = development_balanced_accuracy(
        train_features, train_labels, train_subjects
    )
    predictions = model().fit(train_features, train_labels).predict(test_features)
    return {
        "development": {"balanced_accuracy": development},
        "official_test": {
            "balanced_accuracy": float(
                balanced_accuracy_score(test_labels, predictions)
            ),
            "macro_f1": float(f1_score(test_labels, predictions, average="macro")),
        },
    }
