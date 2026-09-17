"""Conventional and stable-rank features for activity windows."""

import warnings

import numpy as np
from ripser import ripser
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore", message="Failed to load CUDA backend.*")

from stablebear.persistence import Barcode, barcode_to_stable_rank

from data_loader import SENSOR_NAMES
from topology import (
    joint_persistence_diagrams,
    make_coupled_distance_matrix,
    make_joint_point_cloud,
    make_point_cloud,
    persistence_diagrams,
)


STABLE_RANK_GRID = np.linspace(0.0, 2.0, 64)
GRID_STEP = float(STABLE_RANK_GRID[1] - STABLE_RANK_GRID[0])
TOPOLOGICAL_DYNAMICS_NAMES = (
    "H0 mean merge scale",
    "H0 final merge scale",
    "H1 loop count",
    "H1 longest lifetime",
    "H1 total persistence",
    "H1 longest-loop birth",
    "Circular frequency",
    "Circular linearity",
)


def _stable_rank(diagram, grid):
    finite = diagram[np.isfinite(diagram[:, 1])].astype(np.float64)
    if len(finite) == 0:
        return np.zeros_like(grid)
    stable_rank = barcode_to_stable_rank(Barcode(finite))
    return np.asarray(stable_rank(grid), dtype=float)


def _derangement(size, seed):
    target = np.arange(size)
    rng = np.random.default_rng(seed)
    while True:
        permutation = rng.permutation(size)
        if np.all(permutation != target):
            return permutation


def _finite_lifetimes(diagram):
    finite = diagram[np.isfinite(diagram[:, 1])]
    return finite, finite[:, 1] - finite[:, 0]


def _circular_coordinate(cloud, bar, cocycle, coefficient=47):
    """Smooth one persistent cocycle into a circle-valued coordinate."""
    birth, death = bar
    threshold = birth + 0.5 * (death - birth)
    distances = squareform(pdist(cloud))
    sources, targets = np.where(np.triu(distances <= threshold, 1))
    edge_count = len(sources)

    rows = np.repeat(np.arange(edge_count), 2)
    columns = np.column_stack((sources, targets)).reshape(-1)
    values = np.tile((-1.0, 1.0), edge_count)
    incidence = coo_matrix(
        (values, (rows, columns)),
        shape=(edge_count, len(cloud)),
    ).tocsr()

    lifted_cocycle = np.zeros(edge_count)
    edge_indices = {
        (int(source), int(target)): index
        for index, (source, target) in enumerate(zip(sources, targets))
    }
    for raw_source, raw_target, raw_value in cocycle:
        source = int(raw_source)
        target = int(raw_target)
        value = int(raw_value)
        if value > coefficient // 2:
            value -= coefficient
        if source < target:
            edge = (source, target)
            sign = 1
        else:
            edge = (target, source)
            sign = -1
        if edge in edge_indices:
            lifted_cocycle[edge_indices[edge]] = sign * value

    potential = np.zeros(len(cloud))
    potential[1:] = lsqr(
        incidence[:, 1:],
        -lifted_cocycle,
        atol=1e-9,
        btol=1e-9,
        iter_lim=8 * len(cloud),
    )[0]
    return np.mod(potential, 1.0)


def circular_phase_features(
    cloud,
    h1_diagram,
    h1_cocycles,
    *,
    sampling_rate=50.0,
    coefficient=47,
):
    """Return frequency and linearity of the most persistent H1 coordinate."""
    if len(h1_diagram) == 0:
        return 0.0, 0.0

    lifetimes = h1_diagram[:, 1] - h1_diagram[:, 0]
    longest = int(np.argmax(lifetimes))
    coordinate = _circular_coordinate(
        cloud,
        h1_diagram[longest],
        h1_cocycles[longest],
        coefficient,
    )
    phase = np.unwrap(2.0 * np.pi * coordinate)
    time = np.arange(len(phase)) / sampling_rate
    slope, intercept = np.polyfit(time, phase, 1)
    fitted = slope * time + intercept
    total_variation = np.sum((phase - phase.mean()) ** 2)
    if total_variation == 0:
        return 0.0, 0.0

    linearity = 1.0 - np.sum((phase - fitted) ** 2) / total_variation
    frequency = abs(slope) / (2.0 * np.pi)
    return float(frequency), float(np.clip(linearity, 0.0, 1.0))


def topological_dynamics_features(
    signals,
    m: int,
    lag: int,
    *,
    normalize=True,
    sampling_rate=50.0,
    coefficient=47,
    progress=False,
):
    """Summarize connectivity, loops, and circular phase in each window."""
    features = np.empty((len(signals), len(TOPOLOGICAL_DYNAMICS_NAMES)))
    scales = np.empty((len(signals), 1))

    for index, window in enumerate(signals):
        cloud, scales[index, 0] = make_point_cloud(window, m, lag, normalize)
        result = ripser(
            cloud,
            maxdim=1,
            do_cocycles=True,
            coeff=coefficient,
        )
        h0, h1 = result["dgms"]
        finite_h0, h0_lifetimes = _finite_lifetimes(h0)
        finite_h1_mask = np.isfinite(h1[:, 1])
        finite_h1 = h1[finite_h1_mask]
        h1_lifetimes = finite_h1[:, 1] - finite_h1[:, 0]
        finite_h1_cocycles = [
            cocycle
            for cocycle, keep in zip(result["cocycles"][1], finite_h1_mask)
            if keep
        ]
        frequency, linearity = circular_phase_features(
            cloud,
            finite_h1,
            finite_h1_cocycles,
            sampling_rate=sampling_rate,
            coefficient=coefficient,
        )

        if len(finite_h1):
            longest = int(np.argmax(h1_lifetimes))
            h1_max = h1_lifetimes[longest]
            h1_birth = finite_h1[longest, 0]
        else:
            h1_max = 0.0
            h1_birth = 0.0

        features[index] = (
            h0_lifetimes.mean(),
            h0_lifetimes.max(),
            len(finite_h1),
            h1_max,
            h1_lifetimes.sum(),
            h1_birth,
            frequency,
            linearity,
        )

        if progress and (index + 1) % 500 == 0:
            print(f"  processed {index + 1}/{len(signals)} windows", flush=True)

    return features, scales


def multi_sensor_topological_dynamics(sensor_data, m: int, lag: int, **kwargs):
    """Return interpretable topology and phase features for both sensors."""
    feature_blocks = []
    scales = []
    for sensor in SENSOR_NAMES:
        if kwargs.get("progress"):
            print(f"Computing {sensor} topology and phase", flush=True)
        sensor_features, sensor_scales = topological_dynamics_features(
            sensor_data[sensor], m, lag, **kwargs
        )
        feature_blocks.append(sensor_features)
        scales.append(sensor_scales)
    return np.column_stack(feature_blocks), np.column_stack(scales)


def topological_dynamics_feature_names():
    """Return feature names in the same order as the two-sensor matrix."""
    names = []
    for sensor in ("Acceleration", "Gyroscope"):
        names.extend(f"{sensor} {name}" for name in TOPOLOGICAL_DYNAMICS_NAMES)
    return tuple(names)


def stable_rank_features(
    signals,
    m: int,
    lag: int,
    *,
    normalize: bool = True,
    metric: str = "euclidean",
    shuffle_time: bool = False,
    seed: int = 2956,
    grid=STABLE_RANK_GRID,
    progress: bool = False,
):
    """Return sampled H0/H1 stable ranks and point-cloud scales."""
    curves = np.empty((len(signals), 2 * len(grid)))
    scales = np.empty((len(signals), 1))

    for index, window in enumerate(signals):
        if shuffle_time:
            permutation = np.random.default_rng(seed + index).permutation(len(window))
            window = window[permutation]

        _, scales[index, 0] = make_point_cloud(window, m, lag, normalize=False)
        h0, h1 = persistence_diagrams(window, m, lag, normalize, metric)
        curves[index] = np.concatenate((_stable_rank(h0, grid), _stable_rank(h1, grid)))

        if progress and (index + 1) % 500 == 0:
            print(f"  processed {index + 1}/{len(signals)} windows", flush=True)

    return curves, scales


def multi_sensor_stable_ranks(sensor_data, m: int, lag: int, **kwargs):
    """Return [acc H0 | acc H1 | gyro H0 | gyro H1] stable-rank samples."""
    curves = []
    scales = []
    for sensor in SENSOR_NAMES:
        if kwargs.get("progress"):
            print(f"Computing {sensor} stable ranks", flush=True)
        sensor_curves, sensor_scales = stable_rank_features(
            sensor_data[sensor], m, lag, **kwargs
        )
        curves.append(sensor_curves)
        scales.append(sensor_scales)
    return np.concatenate(curves, axis=1), np.concatenate(scales, axis=1)


def joint_sensor_stable_ranks(
    sensor_data,
    m: int,
    lag: int,
    *,
    acceleration_weight: float = 0.5,
    metric: str = "euclidean",
    desynchronize: bool = False,
    seed: int = 2956,
    grid=STABLE_RANK_GRID,
    progress: bool = False,
):
    """Return H0/H1 stable ranks of joint sensor clouds.

    When desynchronizing, a reproducible derangement changes only which
    gyroscope delay point is paired with each acceleration delay point.
    """
    acceleration = sensor_data["body_acc"]
    gyroscope = sensor_data["body_gyro"]
    if len(acceleration) != len(gyroscope):
        raise ValueError("acceleration and gyroscope must contain the same windows")

    curves = np.empty((len(acceleration), 2 * len(grid)))
    scales = np.empty((len(acceleration), 2))

    for index, (acceleration_window, gyroscope_window) in enumerate(
        zip(acceleration, gyroscope)
    ):
        gyroscope_permutation = None
        if desynchronize:
            number_of_points = len(acceleration_window) - (m - 1) * lag
            gyroscope_permutation = _derangement(number_of_points, seed + index)

        _, scales[index] = make_joint_point_cloud(
            acceleration_window,
            gyroscope_window,
            m,
            lag,
            acceleration_weight,
            gyroscope_permutation,
        )
        h0, h1 = joint_persistence_diagrams(
            acceleration_window,
            gyroscope_window,
            m,
            lag,
            acceleration_weight,
            metric,
            gyroscope_permutation,
        )
        curves[index] = np.concatenate((_stable_rank(h0, grid), _stable_rank(h1, grid)))

        if progress and (index + 1) % 500 == 0:
            print(f"  processed {index + 1}/{len(acceleration)} windows", flush=True)

    return curves, scales


def coupled_sensor_stable_ranks(
    sensor_data,
    m: int,
    lag: int,
    *,
    acceleration_weight: float = 0.5,
    desynchronize: bool = False,
    seed: int = 2956,
    grid=STABLE_RANK_GRID,
    progress: bool = False,
):
    """Return stable ranks of the weighted max-coupled sensor filtration."""
    acceleration = sensor_data["body_acc"]
    gyroscope = sensor_data["body_gyro"]
    if len(acceleration) != len(gyroscope):
        raise ValueError("acceleration and gyroscope must contain the same windows")

    curves = np.empty((len(acceleration), 2 * len(grid)))
    scales = np.empty((len(acceleration), 2))

    for index, (acceleration_window, gyroscope_window) in enumerate(
        zip(acceleration, gyroscope)
    ):
        gyroscope_permutation = None
        if desynchronize:
            number_of_points = len(acceleration_window) - (m - 1) * lag
            gyroscope_permutation = _derangement(number_of_points, seed + index)

        distances, scales[index] = make_coupled_distance_matrix(
            acceleration_window,
            gyroscope_window,
            m,
            lag,
            acceleration_weight,
            gyroscope_permutation,
        )
        h0, h1 = ripser(distances, distance_matrix=True, maxdim=1)["dgms"]
        curves[index] = np.concatenate((_stable_rank(h0, grid), _stable_rank(h1, grid)))

        if progress and (index + 1) % 500 == 0:
            print(f"  processed {index + 1}/{len(acceleration)} windows", flush=True)

    return curves, scales


def stable_rank_blocks(features):
    """Name the sensor and homology blocks in the 256-column feature matrix."""
    width = len(STABLE_RANK_GRID)
    if features.shape[1] != 4 * width:
        raise ValueError(f"expected {4 * width} stable-rank columns")

    acceleration_h0 = features[:, :width]
    acceleration_h1 = features[:, width : 2 * width]
    gyroscope_h0 = features[:, 2 * width : 3 * width]
    gyroscope_h1 = features[:, 3 * width :]

    return {
        "Acceleration H0": acceleration_h0,
        "Acceleration H1": acceleration_h1,
        "Gyroscope H0": gyroscope_h0,
        "Gyroscope H1": gyroscope_h1,
        "Both sensors H0": np.column_stack((acceleration_h0, gyroscope_h0)),
        "Both sensors H1": np.column_stack((acceleration_h1, gyroscope_h1)),
        "Acceleration H0+H1": np.column_stack((acceleration_h0, acceleration_h1)),
        "Gyroscope H0+H1": np.column_stack((gyroscope_h0, gyroscope_h1)),
        "All H0+H1": features,
    }


def stable_rank_l1(first, second):
    """Left-Riemann approximation of the stable-rank distance on [0, 2]."""
    differences = np.abs(first - second)
    return GRID_STEP * differences[..., :-1].sum(axis=-1)


def statistical_features(sensor_data):
    """Compute a compact, non-topological baseline from each sensor axis."""
    blocks = []
    for sensor in SENSOR_NAMES:
        windows = sensor_data[sensor]
        axis_features = [
            windows.mean(axis=1),
            windows.std(axis=1),
            windows.min(axis=1),
            windows.max(axis=1),
            np.median(windows, axis=1),
            np.quantile(windows, 0.25, axis=1),
            np.quantile(windows, 0.75, axis=1),
            np.sqrt(np.mean(windows**2, axis=1)),
            np.mean(np.abs(np.diff(windows, axis=1)), axis=1),
        ]

        axis_pairs = ((0, 1), (0, 2), (1, 2))
        correlations = np.stack([
            [np.corrcoef(window[:, i], window[:, j])[0, 1] for i, j in axis_pairs]
            for window in windows
        ])
        magnitude = np.linalg.norm(windows, axis=2)
        magnitude_features = np.column_stack(
            (
                magnitude.mean(axis=1),
                magnitude.std(axis=1),
                np.sqrt(np.mean(magnitude**2, axis=1)),
            )
        )
        blocks.extend(axis_features)
        blocks.extend((correlations, magnitude_features))

    return np.concatenate(blocks, axis=1)
