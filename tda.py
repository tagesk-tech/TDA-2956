import numpy as np


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
    weight = float(acceleration_weight)
    acceleration, acceleration_radius = normalize_cloud(
        delay_cloud(acceleration_window, m=m, lag=lag)
    )
    gyroscope, gyroscope_radius = normalize_cloud(
        delay_cloud(gyroscope_window, m=m, lag=lag)
    )
    cloud = np.column_stack((
        np.sqrt(weight) * acceleration,
        np.sqrt(1.0 - weight) * gyroscope,
    ))
    return cloud, np.asarray((acceleration_radius, gyroscope_radius))


def mean_autocorrelation(windows, max_lag=40):
    centered = windows - windows.mean(axis=1, keepdims=True)
    energy = np.sum(centered**2, axis=1)
    values = []
    for lag in range(max_lag + 1):
        numerator = np.sum(
            centered[:, : windows.shape[1] - lag] * centered[:, lag:], axis=1
        )
        correlation = np.divide(
            numerator,
            energy,
            out=np.full_like(numerator, np.nan),
            where=energy > 0,
        )
        values.append(np.nanmean(correlation))
    return np.asarray(values)


def stable_rank(diagram, grid=GRID):
    diagram = np.asarray(diagram)
    finite = diagram[np.isfinite(diagram[:, 1])]
    lifetimes = finite[:, 1] - finite[:, 0]
    return np.sum(lifetimes[:, None] > grid, axis=0)
