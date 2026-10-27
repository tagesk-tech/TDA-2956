"""Load the UCI Human Activity Recognition sensor windows."""

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np


URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
DATA_DIR = Path(__file__).parent / "data"
SENSOR_FILE = DATA_DIR / "har_sensor_windows.npz"
SENSORS = ("body_acc", "body_gyro")
ACTIVITIES = {
    1: "Walking",
    2: "Walking upstairs",
    3: "Walking downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying",
}


def _safe_extract(zip_path, destination):
    destination = Path(destination).resolve()
    with ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe archive path: {member.filename}")
        archive.extractall(destination)


def dataset_root(data_dir=DATA_DIR):
    """Return the dataset directory, downloading it only when absent."""
    data_dir = Path(data_dir)
    existing = next(
        (path for path in data_dir.rglob("UCI HAR Dataset")
         if (path / "train").is_dir()),
        None,
    )
    if existing is not None:
        return existing

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "uci_har.zip"
    if not archive.exists():
        print("Downloading UCI HAR data", flush=True)
        urlretrieve(URL, archive)
    _safe_extract(archive, data_dir)
    for nested in data_dir.rglob("*.zip"):
        if nested != archive:
            _safe_extract(nested, data_dir)

    root = next(
        (path for path in data_dir.rglob("UCI HAR Dataset")
         if (path / "train").is_dir()),
        None,
    )
    if root is None:
        raise FileNotFoundError("UCI HAR Dataset was not found in the archive")
    return root


def load_split(split, data_dir=DATA_DIR):
    """Return sensor arrays, labels, and subject IDs for one supplied split."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")

    sensor_file = Path(data_dir) / SENSOR_FILE.name
    if sensor_file.exists():
        with np.load(sensor_file, allow_pickle=False) as saved:
            sensors = {
                sensor: saved[f"{split}_{sensor}"]
                for sensor in SENSORS
            }
            labels = saved[f"{split}_labels"]
            subjects = saved[f"{split}_subjects"]
        validate_split(sensors, labels, subjects)
        return sensors, labels, subjects

    directory = dataset_root(data_dir) / split
    inertial = directory / "Inertial Signals"

    sensors = {}
    for sensor in SENSORS:
        axes = [
            np.loadtxt(inertial / f"{sensor}_{axis}_{split}.txt")
            for axis in "xyz"
        ]
        sensors[sensor] = np.stack(axes, axis=-1)

    labels = np.loadtxt(directory / f"y_{split}.txt", dtype=int)
    subjects = np.loadtxt(directory / f"subject_{split}.txt", dtype=int)
    validate_split(sensors, labels, subjects)
    return sensors, labels, subjects


def load_engineered_features(split, data_dir=DATA_DIR):
    root = dataset_root(data_dir) / split
    return np.loadtxt(root / f"X_{split}.txt")


def validate_split(sensors, labels, subjects):
    n = len(labels)
    if labels.shape != (n,) or subjects.shape != (n,):
        raise ValueError("labels and subjects must be one-dimensional")
    if set(np.unique(labels)) != set(ACTIVITIES):
        raise ValueError("expected activity labels 1 through 6")
    for name in SENSORS:
        if sensors[name].shape != (n, 128, 3):
            raise ValueError(f"unexpected {name} shape: {sensors[name].shape}")
        if not np.isfinite(sensors[name]).all():
            raise ValueError(f"{name} contains non-finite values")


def first_subject_activity_windows(labels, subjects, per_group=1):
    """Indices for a small, subject-balanced real-data smoke run."""
    selected = []
    for subject in np.unique(subjects):
        for activity in ACTIVITIES:
            matches = np.flatnonzero((subjects == subject) & (labels == activity))
            selected.extend(matches[:per_group])
    return np.asarray(selected, dtype=int)
