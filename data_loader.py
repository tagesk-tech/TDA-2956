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


if __name__ == "__main__":
    x, y, subjects = load_data("train")
    print("Acceleration windows:", x.shape)
    print("Labels:", y.shape)
    print("Training subjects:", np.unique(subjects))
