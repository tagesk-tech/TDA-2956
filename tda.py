"""The complete topological feature map used by the analysis."""

import warnings

import numpy as np
from ripser import ripser
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore", message="Failed to load CUDA backend.*")

from stablebear.persistence import Barcode, barcode_to_stable_rank

from data import SENSORS


GRID = np.linspace(0.0, 2.0, 64)
GRID_STEP = float(GRID[1] - GRID[0])
DYNAMICS_NAMES = (
    "H0 mean merge scale",
    "H0 final merge scale",
    "H1 loop count",
    "H1 longest lifetime",
    "H1 total persistence",
    "H1 longest-loop birth",
    "Circular frequency",
    "Circular linearity",
)


def delay_cloud(window, m=3, lag=15):
    """Map a (samples, channels) path to overlapping delay vectors."""
    window = np.asarray(window, dtype=float)
    if window.ndim != 2:
        raise ValueError("window must have shape (samples, channels)")
    if m < 1 or lag < 1:
        raise ValueError("m and lag must be positive")
    count = len(window) - (m - 1) * lag
    if count <= 0:
        raise ValueError("m and lag are too large for the window")
    indices = np.arange(count)[:, None] + lag * np.arange(m)
    return window[indices].reshape(count, -1)


def normalize_cloud(cloud):
    """Center a cloud and divide by its RMS radius."""
    centered = cloud - cloud.mean(axis=0)
    radius = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if radius == 0:
        raise ValueError("cannot normalize a constant point cloud")
    return centered / radius, radius


def joint_delay_cloud(acceleration_window, gyroscope_window,
                      acceleration_weight=0.5, m=3, lag=15):
    """Join separately normalized, synchronized sensor delay clouds."""
    weight = float(acceleration_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("acceleration_weight must lie between 0 and 1")
    acceleration, acceleration_radius = normalize_cloud(
        delay_cloud(acceleration_window, m=m, lag=lag)
    )
    gyroscope, gyroscope_radius = normalize_cloud(
        delay_cloud(gyroscope_window, m=m, lag=lag)
    )
    cloud = np.column_stack((np.sqrt(weight) * acceleration,
                             np.sqrt(1.0 - weight) * gyroscope))
    return cloud, np.asarray((acceleration_radius, gyroscope_radius))


def mean_autocorrelation(windows, max_lag=40):
    """Mean normalized autocorrelation over windows and sensor axes."""
    centered = windows - windows.mean(axis=1, keepdims=True)
    energy = np.sum(centered**2, axis=1)
    values = []
    for lag in range(max_lag + 1):
        numerator = np.sum(
            centered[:, : windows.shape[1] - lag] * centered[:, lag:], axis=1
        )
        correlation = np.divide(
            numerator, energy, out=np.full_like(numerator, np.nan), where=energy > 0
        )
        values.append(np.nanmean(correlation))
    return np.asarray(values)


def first_nonpositive_lag(autocorrelation):
    candidates = np.flatnonzero(np.asarray(autocorrelation)[1:] <= 0) + 1
    if len(candidates) == 0:
        raise ValueError("autocorrelation does not cross zero")
    return int(candidates[0])


def stable_rank(diagram, grid=GRID):
    """Evaluate the lifetime stable rank of a persistence diagram."""
    finite = np.asarray(diagram)[np.isfinite(diagram[:, 1])].astype(np.float64)
    if len(finite) == 0:
        return np.zeros_like(grid)
    function = barcode_to_stable_rank(Barcode(finite))
    return np.asarray(function(grid), dtype=float)


def stable_rank_distance(first, second, grid_step=GRID_STEP):
    """Blockwise left-Riemann L1 distance for 64-coordinate stable ranks."""
    first, second = np.broadcast_arrays(first, second)
    if first.shape[-1] % len(GRID):
        raise ValueError("stable-rank arrays must contain complete 64-column blocks")
    blocks = first.shape[-1] // len(GRID)
    difference = np.abs(first - second).reshape(
        *first.shape[:-1], blocks, len(GRID)
    )
    return grid_step * difference[..., :-1].sum(axis=(-1, -2))


def _finite(diagram):
    finite = diagram[np.isfinite(diagram[:, 1])]
    return finite, finite[:, 1] - finite[:, 0]


def _circular_coordinate(cloud, bar, cocycle, coefficient):
    """Smooth a persistent cocycle into a circle-valued vertex coordinate."""
    birth, death = bar
    threshold = birth + 0.5 * (death - birth)
    distances = squareform(pdist(cloud))
    sources, targets = np.where(np.triu(distances <= threshold, 1))

    edge_count = len(sources)
    rows = np.repeat(np.arange(edge_count), 2)
    columns = np.column_stack((sources, targets)).reshape(-1)
    values = np.tile((-1.0, 1.0), edge_count)
    incidence = coo_matrix(
        (values, (rows, columns)), shape=(edge_count, len(cloud))
    ).tocsr()

    lifted = np.zeros(edge_count)
    edge_index = {
        (int(source), int(target)): index
        for index, (source, target) in enumerate(zip(sources, targets))
    }
    for raw_source, raw_target, raw_value in cocycle:
        source, target, value = int(raw_source), int(raw_target), int(raw_value)
        if value > coefficient // 2:
            value -= coefficient
        edge = (min(source, target), max(source, target))
        if edge in edge_index:
            lifted[edge_index[edge]] = value if source < target else -value

    solution = np.asarray(
        lsqr(
            incidence[:, 1:], -lifted, atol=1e-9, btol=1e-9,
            iter_lim=8 * len(cloud),
        )[0],
        dtype=float,
    )
    if solution.shape != (len(cloud) - 1,):
        raise RuntimeError("circular-coordinate solve returned the wrong shape")
    return np.mod(np.concatenate(([0.0], solution)), 1.0)


def _phase_features(cloud, diagram, cocycles, sampling_rate, coefficient):
    if len(diagram) == 0:
        return 0.0, 0.0
    lifetimes = diagram[:, 1] - diagram[:, 0]
    longest = int(np.argmax(lifetimes))
    coordinate = _circular_coordinate(
        cloud, diagram[longest], cocycles[longest], coefficient
    )
    phase = np.unwrap(2 * np.pi * coordinate)
    time = np.arange(len(phase)) / sampling_rate
    slope, intercept = np.polyfit(time, phase, 1)
    fitted = slope * time + intercept
    total = np.sum((phase - phase.mean()) ** 2)
    if total == 0:
        return 0.0, 0.0
    linearity = 1 - np.sum((phase - fitted) ** 2) / total
    return abs(float(slope)) / (2 * np.pi), float(np.clip(linearity, 0, 1))


def diagram_dynamics(cloud, result, sampling_rate=50.0, coefficient=47):
    """Six diagram summaries plus circular frequency and phase linearity."""
    h0, h1 = result["dgms"]
    _, h0_lifetimes = _finite(h0)
    h1_mask = np.isfinite(h1[:, 1])
    finite_h1 = h1[h1_mask]
    h1_lifetimes = finite_h1[:, 1] - finite_h1[:, 0]
    h1_cocycles = [
        cocycle for cocycle, keep in zip(result["cocycles"][1], h1_mask) if keep
    ]
    frequency, linearity = _phase_features(
        cloud, finite_h1, h1_cocycles, sampling_rate, coefficient
    )
    if len(finite_h1):
        longest = int(np.argmax(h1_lifetimes))
        longest_lifetime = h1_lifetimes[longest]
        longest_birth = finite_h1[longest, 0]
    else:
        longest_lifetime = longest_birth = 0.0
    return np.asarray((
        h0_lifetimes.mean(), h0_lifetimes.max(), len(finite_h1),
        longest_lifetime, h1_lifetimes.sum(), longest_birth,
        frequency, linearity,
    ))


def feature_matrices(
    sensor_data,
    m=3,
    lag=15,
    *,
    shuffle_seed=None,
    include_dynamics=False,
    progress=False,
):
    """Compute both sensors, reusing delay persistence for dynamics when requested."""
    n = len(sensor_data[SENSORS[0]])
    ranks = np.empty((n, 4 * len(GRID)))
    scales = np.empty((n, 2))
    dynamics = np.empty((n, 2 * len(DYNAMICS_NAMES))) if include_dynamics else None

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
            result = ripser(
                cloud, maxdim=1, do_cocycles=include_dynamics, coeff=47
            )
            start = sensor_index * 2 * len(GRID)
            ranks[index, start:start + len(GRID)] = stable_rank(result["dgms"][0])
            ranks[index, start + len(GRID):start + 2 * len(GRID)] = stable_rank(
                result["dgms"][1]
            )
            if include_dynamics:
                width = len(DYNAMICS_NAMES)
                dynamics[index, sensor_index * width:(sensor_index + 1) * width] = (
                    diagram_dynamics(cloud, result)
                )
            if progress and (index + 1) % 500 == 0:
                print(f"{sensor}: {index + 1}/{n}", flush=True)
    return ranks, scales, dynamics


def joint_feature_matrices(sensor_data, acceleration_weights, m=3, lag=15,
                           *, progress=False):
    """Stable ranks of weighted joint acceleration-gyroscope delay clouds."""
    weights = np.atleast_1d(np.asarray(acceleration_weights, dtype=float))
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("acceleration_weights must be a non-empty vector")
    if np.any((weights < 0.0) | (weights > 1.0)):
        raise ValueError("acceleration weights must lie between 0 and 1")

    n = len(sensor_data["body_acc"])
    if len(sensor_data["body_gyro"]) != n:
        raise ValueError("sensor arrays must contain the same number of windows")
    ranks = np.empty((len(weights), n, 2 * len(GRID)))
    scales = np.empty((n, 2))

    paired = zip(sensor_data["body_acc"], sensor_data["body_gyro"])
    for index, (acceleration, gyroscope) in enumerate(paired):
        for weight_index, weight in enumerate(weights):
            cloud, radii = joint_delay_cloud(
                acceleration, gyroscope, weight, m=m, lag=lag
            )
            if weight_index == 0:
                scales[index] = radii
            result = ripser(cloud, maxdim=1, coeff=47)
            ranks[weight_index, index, :len(GRID)] = stable_rank(
                result["dgms"][0]
            )
            ranks[weight_index, index, len(GRID):] = stable_rank(
                result["dgms"][1]
            )
        if progress and (index + 1) % 500 == 0:
            print(f"joint sensors: {index + 1}/{n}", flush=True)
    return ranks, scales
