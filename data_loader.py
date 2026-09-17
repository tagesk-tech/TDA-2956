"""Download and load the UCI smartphone human-activity dataset."""

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np


DATA_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
DATA_DIR = Path(__file__).resolve().parent / "data"
SENSOR_NAMES = ("body_acc", "body_gyro")
SPLITS = ("train", "test")
ACTIVITY_NAMES = {
    1: "Walking",
    2: "Walking upstairs",
    3: "Walking downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying",
}


def _safe_extract(zip_path: Path, destination: Path) -> None:
    """Extract a ZIP file while rejecting paths outside the data directory."""
    destination = destination.resolve()
    with ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        archive.extractall(destination)


def download_data(data_dir: Path = DATA_DIR) -> Path:
    """Download the dataset if necessary and return its root directory."""
    data_dir = Path(data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    existing = next(
        (p for p in data_dir.rglob("UCI HAR Dataset") if (p / "train").is_dir()),
        None,
    )
    if existing is not None:
        return existing

    outer_zip = data_dir / "uci_har.zip"
    if not outer_zip.exists():
        print("Downloading the UCI HAR dataset...")
        urlretrieve(DATA_URL, outer_zip)

    _safe_extract(outer_zip, data_dir)
    for nested_zip in data_dir.rglob("*.zip"):
        if nested_zip != outer_zip:
            _safe_extract(nested_zip, data_dir)

    dataset_root = next(
        (p for p in data_dir.rglob("UCI HAR Dataset") if (p / "train").is_dir()),
        None,
    )
    if dataset_root is None:
        raise FileNotFoundError("The archive did not contain the expected dataset")
    return dataset_root


def _split_paths(split, data_dir):
    if split not in SPLITS:
        raise ValueError("split must be 'train' or 'test'")
    root = download_data(data_dir)
    split_dir = root / split
    return split_dir, split_dir / "Inertial Signals"


def _load_sensor(signal_dir, sensor, split):
    axes = [
        np.loadtxt(signal_dir / f"{sensor}_{axis}_{split}.txt")
        for axis in ("x", "y", "z")
    ]
    return np.stack(axes, axis=-1)


def _load_labels_and_subjects(split_dir, split):
    labels = np.loadtxt(split_dir / f"y_{split}.txt", dtype=int)
    subjects = np.loadtxt(split_dir / f"subject_{split}.txt", dtype=int)
    return labels, subjects


def load_data(split: str = "train", data_dir: Path = DATA_DIR):
    """Return acceleration windows, activity labels, and subject IDs."""
    split_dir, signal_dir = _split_paths(split, data_dir)
    acceleration = _load_sensor(signal_dir, "body_acc", split)
    labels, subjects = _load_labels_and_subjects(split_dir, split)
    return acceleration, labels, subjects


def load_inertial_data(split: str = "train", data_dir: Path = DATA_DIR):
    """Return body-acceleration and gyroscope windows keyed by sensor name."""
    split_dir, signal_dir = _split_paths(split, data_dir)
    signals = {
        sensor: _load_sensor(signal_dir, sensor, split)
        for sensor in SENSOR_NAMES
    }
    labels, subjects = _load_labels_and_subjects(split_dir, split)
    return signals, labels, subjects


def load_engineered_features(split: str = "train", data_dir: Path = DATA_DIR):
    """Load the 561-feature reference representation supplied with UCI HAR."""
    split_dir, _ = _split_paths(split, data_dir)
    return np.loadtxt(split_dir / f"X_{split}.txt")


def check_data(signals, labels, subjects) -> None:
    """Validate one dataset split and print a compact summary."""
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

    print("Signals:", signals.shape)
    print("Labels:", labels.shape)
    print("Subjects:", subjects.shape)
    print("Number of subjects:", len(np.unique(subjects)))
    activities, counts = np.unique(labels, return_counts=True)
    for activity, count in zip(activities, counts):
        print(f"{ACTIVITY_NAMES[activity]}: {count}")


if __name__ == "__main__":
    x, y, subjects = load_data("train")
    print("Acceleration windows:", x.shape)
    print("Labels:", y.shape)
    print("Training subjects:", np.unique(subjects))
