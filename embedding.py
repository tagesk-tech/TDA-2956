"""Time-delay representations for sensor signals."""

import numpy as np


def delay_embedding(signal, m: int, lag: int):
    """Return overlapping delay vectors from a sampled signal."""
    signal = np.asarray(signal)

    if signal.ndim == 1:
        signal = signal[:, None]
    if signal.ndim != 2:
        raise ValueError("signal must have shape (samples, channels)")
    if m < 1 or lag < 1:
        raise ValueError("m and lag must be positive")

    number_of_points = len(signal) - (m - 1) * lag
    if number_of_points <= 0:
        raise ValueError("m and lag are too large for this signal")

    indices = np.arange(number_of_points)[:, None] + lag * np.arange(m)
    return signal[indices].reshape(number_of_points, -1)


def joint_delay_embedding(acceleration, gyroscope, m: int, lag: int):
    """Join synchronized acceleration and gyroscope delay vectors."""
    acceleration_cloud = delay_embedding(acceleration, m, lag)
    gyroscope_cloud = delay_embedding(gyroscope, m, lag)

    if acceleration_cloud.shape[0] != gyroscope_cloud.shape[0]:
        raise ValueError("acceleration and gyroscope windows must have equal length")

    return np.column_stack((acceleration_cloud, gyroscope_cloud))


def mean_autocorrelation(signals, max_lag: int):
    """Average normalized autocorrelation over windows and channels."""
    signals = np.asarray(signals)
    centered = signals - signals.mean(axis=1, keepdims=True)
    energy = np.sum(centered**2, axis=1)

    autocorrelation = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        left = centered[:, : len(centered[0]) - lag]
        right = centered[:, lag:]
        numerator = np.sum(left * right, axis=1)
        correlations = np.divide(
            numerator,
            energy,
            out=np.full_like(numerator, np.nan),
            where=energy > 0,
        )
        autocorrelation[lag] = correlations[np.isfinite(correlations)].mean()

    return autocorrelation


def first_nonpositive_lag(autocorrelation):
    """Return the first lag at which the autocorrelation is non-positive."""
    candidates = np.flatnonzero(np.asarray(autocorrelation)[1:] <= 0) + 1
    if len(candidates) == 0:
        raise ValueError("autocorrelation does not cross zero")
    return int(candidates[0])
