import numpy as np
from ripser import ripser

from data import SENSORS


GRID = np.linspace(0.0, 2.0, 64)


def delay_cloud(window, m=3, lag=15):
    count = len(window) - (m - 1) * lag
    indices = np.arange(count)[:, None] + lag * np.arange(m)
    return window[indices].reshape(count, -1)


def normalize_cloud(cloud):
    centered = cloud - cloud.mean(axis=0)
    radius = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    return centered / radius, radius


def joint_delay_cloud(
    acceleration_window,
    gyroscope_window,
    acceleration_weight=0.5,
    m=3,
    lag=15,
):
    acceleration, acceleration_radius = normalize_cloud(
        delay_cloud(acceleration_window, m, lag)
    )
    gyroscope, gyroscope_radius = normalize_cloud(
        delay_cloud(gyroscope_window, m, lag)
    )
    cloud = np.column_stack((
        np.sqrt(acceleration_weight) * acceleration,
        np.sqrt(1.0 - acceleration_weight) * gyroscope,
    ))
    return cloud, np.asarray((acceleration_radius, gyroscope_radius))


def mean_autocorrelation(windows, max_lag=40):
    centered = windows - windows.mean(axis=1, keepdims=True)
    energy = np.sum(centered**2, axis=1)
    return np.asarray([
        np.mean(np.sum(
            centered[:, :windows.shape[1] - lag] * centered[:, lag:],
            axis=1,
        ) / energy)
        for lag in range(max_lag + 1)
    ])


def first_nonpositive_lag(autocorrelation):
    return int(np.flatnonzero(np.asarray(autocorrelation)[1:] <= 0)[0] + 1)


def stable_rank(diagram, grid=GRID):
    finite = np.asarray(diagram)[np.isfinite(diagram[:, 1])]
    lifetimes = finite[:, 1] - finite[:, 0]
    return np.sum(lifetimes[:, None] > grid, axis=0)


def feature_matrices(
    sensor_data,
    m=3,
    lag=15,
    *,
    shuffle_seed=None,
    progress=False,
):
    n = len(sensor_data[SENSORS[0]])
    ranks = np.empty((n, 4 * len(GRID)))
    scales = np.empty((n, 2))

    for sensor_index, sensor in enumerate(SENSORS):
        for index, original in enumerate(sensor_data[sensor]):
            window = original
            if shuffle_seed is not None:
                permutation = np.random.default_rng(
                    shuffle_seed + index
                ).permutation(len(window))
                window = window[permutation]
            cloud, scales[index, sensor_index] = normalize_cloud(
                delay_cloud(window, m, lag)
            )
            diagrams = ripser(cloud, maxdim=1, coeff=47)["dgms"]
            start = sensor_index * 2 * len(GRID)
            ranks[index, start:start + len(GRID)] = stable_rank(diagrams[0])
            ranks[index, start + len(GRID):start + 2 * len(GRID)] = stable_rank(
                diagrams[1]
            )
            if progress and (index + 1) % 500 == 0:
                print(f"{sensor}: {index + 1}/{n}", flush=True)
    return ranks, scales


def joint_feature_matrices(
    sensor_data,
    acceleration_weights,
    m=3,
    lag=15,
    *,
    progress=False,
):
    weights = np.atleast_1d(np.asarray(acceleration_weights, dtype=float))
    n = len(sensor_data["body_acc"])
    ranks = np.empty((len(weights), n, 2 * len(GRID)))

    paired = zip(sensor_data["body_acc"], sensor_data["body_gyro"])
    for index, (acceleration, gyroscope) in enumerate(paired):
        for weight_index, weight in enumerate(weights):
            cloud, _ = joint_delay_cloud(
                acceleration, gyroscope, weight, m, lag
            )
            diagrams = ripser(cloud, maxdim=1, coeff=47)["dgms"]
            ranks[weight_index, index, :len(GRID)] = stable_rank(diagrams[0])
            ranks[weight_index, index, len(GRID):] = stable_rank(diagrams[1])
        if progress and (index + 1) % 500 == 0:
            print(f"joint: {index + 1}/{n}", flush=True)
    return ranks
