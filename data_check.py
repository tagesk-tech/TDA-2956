"""Small, reusable checks for the loaded UCI HAR arrays."""

import numpy as np


ACTIVITY_NAMES = {
    1: "Walking",
    2: "Walking upstairs",
    3: "Walking downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying",
}


def check_data(signals, labels, subjects) -> None:
    """Validate one complete dataset split and print a compact summary."""
    number_of_windows = signals.shape[0]
    problems = []

    if signals.shape[1:] != (128, 3):
        problems.append(f"expected signal shape (windows, 128, 3), got {signals.shape}")
    if labels.shape != (number_of_windows,):
        problems.append("labels and signals contain different numbers of windows")
    if subjects.shape != (number_of_windows,):
        problems.append("subject IDs and signals contain different numbers of windows")
    if not np.isfinite(signals).all():
        problems.append("signals contain NaN or infinite values")
    if set(np.unique(labels)) != set(ACTIVITY_NAMES):
        problems.append(f"expected activity labels 1--6, got {np.unique(labels)}")
    if problems:
        raise ValueError("Invalid UCI HAR split:\n- " + "\n- ".join(problems))

    activities, counts = np.unique(labels, return_counts=True)

    print("Signals:", signals.shape)
    print("Labels:", labels.shape)
    print("Subjects:", subjects.shape)
    print("Number of subjects:", len(np.unique(subjects)))

    for activity, count in zip(activities, counts):
        print(f"{ACTIVITY_NAMES[activity]}: {count}")
