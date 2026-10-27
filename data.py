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


def dataset_root(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    root = next(
        (path for path in data_dir.rglob("UCI HAR Dataset")
         if (path / "train").is_dir()),
        None,
    )
    if root is not None:
        return root

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "uci_har.zip"
    if not archive.exists():
        urlretrieve(URL, archive)
    with ZipFile(archive) as zipped:
        zipped.extractall(data_dir)
    for nested in data_dir.glob("*.zip"):
        if nested != archive:
            with ZipFile(nested) as zipped:
                zipped.extractall(data_dir)
    return next(
        path for path in data_dir.rglob("UCI HAR Dataset")
        if (path / "train").is_dir()
    )


def load_split(split, data_dir=DATA_DIR):
    sensor_file = Path(data_dir) / SENSOR_FILE.name
    if sensor_file.exists():
        with np.load(sensor_file, allow_pickle=False) as saved:
            sensors = {
                sensor: saved[f"{split}_{sensor}"]
                for sensor in SENSORS
            }
            labels = saved[f"{split}_labels"]
            subjects = saved[f"{split}_subjects"]
        return sensors, labels, subjects

    directory = dataset_root(data_dir) / split
    inertial = directory / "Inertial Signals"
    sensors = {
        sensor: np.stack([
            np.loadtxt(inertial / f"{sensor}_{axis}_{split}.txt")
            for axis in "xyz"
        ], axis=-1)
        for sensor in SENSORS
    }
    labels = np.loadtxt(directory / f"y_{split}.txt", dtype=int)
    subjects = np.loadtxt(directory / f"subject_{split}.txt", dtype=int)
    return sensors, labels, subjects


def load_engineered_features(split, data_dir=DATA_DIR):
    return np.loadtxt(dataset_root(data_dir) / split / f"X_{split}.txt")
