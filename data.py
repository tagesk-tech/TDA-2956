from pathlib import Path

import numpy as np


DATA_FILE = Path(__file__).parent / "data" / "har_sensor_windows.npz"
ACTIVITIES = {
    1: "Walking",
    2: "Walking upstairs",
    3: "Walking downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying",
}


def load_split(split):
    with np.load(DATA_FILE, allow_pickle=False) as data:
        sensors = {
            sensor: data[f"{split}_{sensor}"]
            for sensor in ("body_acc", "body_gyro")
        }
        labels = data[f"{split}_labels"]
        subjects = data[f"{split}_subjects"]
    return sensors, labels, subjects
