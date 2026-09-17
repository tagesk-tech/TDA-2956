"""Persistent homology of Takens-type sensor point clouds."""

import numpy as np
from ripser import ripser
from scipy.spatial.distance import pdist, squareform

from embedding import delay_embedding, joint_delay_embedding


def _center_and_scale(cloud):
    centered = cloud - cloud.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    return centered, scale


def _ripser_diagrams(cloud, metric):
    if metric == "euclidean":
        return ripser(cloud, maxdim=1)["dgms"]
    if metric == "cityblock":
        distances = squareform(pdist(cloud, metric="cityblock"))
        return ripser(distances, distance_matrix=True, maxdim=1)["dgms"]
    raise ValueError("metric must be 'euclidean' or 'cityblock'")


def _normalized_sensor_blocks(
    acceleration,
    gyroscope,
    m,
    lag,
    gyroscope_permutation=None,
):
    joint = joint_delay_embedding(acceleration, gyroscope, m, lag)
    width = joint.shape[1] // 2
    acceleration_cloud, acceleration_scale = _center_and_scale(joint[:, :width])
    gyroscope_cloud, gyroscope_scale = _center_and_scale(joint[:, width:])

    if acceleration_scale == 0 or gyroscope_scale == 0:
        raise ValueError("cannot normalize a constant sensor cloud")

    acceleration_cloud /= acceleration_scale
    gyroscope_cloud /= gyroscope_scale

    if gyroscope_permutation is not None:
        gyroscope_permutation = np.asarray(gyroscope_permutation)
        expected = np.arange(len(gyroscope_cloud))
        if (
            gyroscope_permutation.shape != expected.shape
            or not np.array_equal(np.sort(gyroscope_permutation), expected)
        ):
            raise ValueError("gyroscope_permutation must permute every cloud index")
        gyroscope_cloud = gyroscope_cloud[gyroscope_permutation]

    scales = np.asarray((acceleration_scale, gyroscope_scale))
    return acceleration_cloud, gyroscope_cloud, scales


def make_point_cloud(window, m: int, lag: int, normalize: bool = False):
    """Build a delay cloud and optionally remove its overall geometric scale."""
    cloud = delay_embedding(window, m, lag)
    centered, scale = _center_and_scale(cloud)

    if normalize:
        if scale == 0:
            raise ValueError("cannot normalize a constant point cloud")
        cloud = centered / scale

    return cloud, scale


def make_joint_point_cloud(
    acceleration,
    gyroscope,
    m: int,
    lag: int,
    acceleration_weight: float = 0.5,
    gyroscope_permutation=None,
):
    """Build a joint cloud with separately normalized sensor blocks.

    The weight controls the acceleration contribution to squared Euclidean
    distance; the gyroscope contribution is one minus this value.
    """
    if not 0 < acceleration_weight < 1:
        raise ValueError("acceleration_weight must lie strictly between 0 and 1")

    acceleration_cloud, gyroscope_cloud, scales = _normalized_sensor_blocks(
        acceleration,
        gyroscope,
        m,
        lag,
        gyroscope_permutation,
    )

    cloud = np.column_stack((
        np.sqrt(acceleration_weight) * acceleration_cloud,
        np.sqrt(1.0 - acceleration_weight) * gyroscope_cloud,
    ))
    return cloud, scales


def make_coupled_distance_matrix(
    acceleration,
    gyroscope,
    m: int,
    lag: int,
    acceleration_weight: float = 0.5,
    gyroscope_permutation=None,
):
    """Return the weighted max-product metric on paired sensor delays.

    At equal weight, two delay points are within epsilon exactly when they
    are within epsilon in both normalized sensor clouds.
    """
    if not 0 < acceleration_weight < 1:
        raise ValueError("acceleration_weight must lie strictly between 0 and 1")

    acceleration_cloud, gyroscope_cloud, scales = _normalized_sensor_blocks(
        acceleration,
        gyroscope,
        m,
        lag,
        gyroscope_permutation,
    )
    acceleration_distances = squareform(pdist(acceleration_cloud))
    gyroscope_distances = squareform(pdist(gyroscope_cloud))
    acceleration_distances *= np.sqrt(2.0 * acceleration_weight)
    gyroscope_distances *= np.sqrt(2.0 * (1.0 - acceleration_weight))
    distances = np.maximum(acceleration_distances, gyroscope_distances)
    return distances, scales


def persistence_diagrams(
    window,
    m: int,
    lag: int,
    normalize: bool = False,
    metric: str = "euclidean",
):
    """Compute the H0 and H1 diagrams of one delay-embedded window."""
    cloud, _ = make_point_cloud(window, m, lag, normalize)

    return _ripser_diagrams(cloud, metric)


def joint_persistence_diagrams(
    acceleration,
    gyroscope,
    m: int,
    lag: int,
    acceleration_weight: float = 0.5,
    metric: str = "euclidean",
    gyroscope_permutation=None,
):
    """Compute H0 and H1 for synchronized acceleration-gyroscope motion."""
    cloud, _ = make_joint_point_cloud(
        acceleration,
        gyroscope,
        m,
        lag,
        acceleration_weight,
        gyroscope_permutation,
    )
    return _ripser_diagrams(cloud, metric)


def coupled_persistence_diagrams(
    acceleration,
    gyroscope,
    m: int,
    lag: int,
    acceleration_weight: float = 0.5,
    gyroscope_permutation=None,
):
    """Compute persistence of the max-coupled sensor filtration."""
    distances, _ = make_coupled_distance_matrix(
        acceleration,
        gyroscope,
        m,
        lag,
        acceleration_weight,
        gyroscope_permutation,
    )
    return ripser(distances, distance_matrix=True, maxdim=1)["dgms"]


def representative_diagrams(
    signals, labels, subjects, subject: int, m: int, lag: int, normalize: bool = False
):
    """Return one persistence calculation per activity for one subject."""
    from data_check import ACTIVITY_NAMES

    diagrams = {}
    for activity in ACTIVITY_NAMES:
        index = np.flatnonzero((subjects == subject) & (labels == activity))[0]
        diagrams[activity] = persistence_diagrams(signals[index], m, lag, normalize)
    return diagrams
