"""Small mathematical and data-contract checks for the analysis pipeline."""

import unittest

import numpy as np

from data_loader import load_inertial_data
from embedding import delay_embedding, joint_delay_embedding
from features import (
    STABLE_RANK_GRID,
    circular_phase_features,
    coupled_sensor_stable_ranks,
    joint_sensor_stable_ranks,
    multi_sensor_topological_dynamics,
    stable_rank_blocks,
    stable_rank_features,
    stable_rank_l1,
)
from ripser import ripser
from run_extensions import subject_activity_prototypes
from run_robustness import _paired_subject_difference
from topology import (
    make_coupled_distance_matrix,
    make_joint_point_cloud,
    make_point_cloud,
    persistence_diagrams,
)


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.signals, cls.labels, cls.subjects = load_inertial_data("train")

    def test_delay_embedding_shape_and_order(self):
        window = self.signals["body_acc"][0]
        cloud = delay_embedding(window, m=3, lag=15)
        self.assertEqual(cloud.shape, (98, 9))
        np.testing.assert_array_equal(cloud[0], window[[0, 15, 30]].reshape(-1))

    def test_joint_delay_embedding_shape_and_order(self):
        acceleration = self.signals["body_acc"][0]
        gyroscope = self.signals["body_gyro"][0]
        cloud = joint_delay_embedding(acceleration, gyroscope, m=3, lag=15)
        self.assertEqual(cloud.shape, (98, 18))
        np.testing.assert_array_equal(
            cloud[0, :9], acceleration[[0, 15, 30]].reshape(-1)
        )
        np.testing.assert_array_equal(
            cloud[0, 9:], gyroscope[[0, 15, 30]].reshape(-1)
        )

    def test_official_split_contract(self):
        test_signals, _, test_subjects = load_inertial_data("test")
        self.assertEqual(self.signals["body_acc"].shape, (7352, 128, 3))
        self.assertEqual(test_signals["body_acc"].shape, (2947, 128, 3))
        self.assertTrue(set(self.subjects).isdisjoint(test_subjects))

    def test_unordered_cloud_is_shuffle_invariant(self):
        window = self.signals["body_acc"][0]
        shuffled = window[np.random.default_rng(2956).permutation(len(window))]
        first = persistence_diagrams(window, 1, 1, normalize=True)
        second = persistence_diagrams(shuffled, 1, 1, normalize=True)
        for left, right in zip(first, second):
            left = left[np.lexsort((left[:, 1], left[:, 0]))]
            right = right[np.lexsort((right[:, 1], right[:, 0]))]
            np.testing.assert_allclose(left, right)

    def test_delay_cloud_is_time_reversal_invariant(self):
        window = self.signals["body_acc"][0]
        first = persistence_diagrams(window, 3, 15, normalize=True)
        second = persistence_diagrams(window[::-1], 3, 15, normalize=True)
        for left, right in zip(first, second):
            left = left[np.lexsort((left[:, 1], left[:, 0]))]
            right = right[np.lexsort((right[:, 1], right[:, 0]))]
            np.testing.assert_allclose(left, right, atol=1e-6)

    def test_normalized_cloud_removes_translation_and_scale(self):
        window = self.signals["body_acc"][0]
        first, _ = make_point_cloud(window, m=3, lag=15, normalize=True)
        second, _ = make_point_cloud(7.0 * window + 4.0, m=3, lag=15, normalize=True)
        np.testing.assert_allclose(first, second, atol=1e-12)

    def test_joint_cloud_balances_sensor_scales(self):
        acceleration = self.signals["body_acc"][0]
        gyroscope = self.signals["body_gyro"][0]
        cloud, scales = make_joint_point_cloud(
            acceleration, gyroscope, m=3, lag=15
        )
        self.assertEqual(cloud.shape, (98, 18))
        self.assertEqual(scales.shape, (2,))
        self.assertAlmostEqual(np.sqrt(np.mean(np.sum(cloud**2, axis=1))), 1.0)
        self.assertAlmostEqual(
            np.mean(np.sum(cloud[:, :9] ** 2, axis=1)), 0.5
        )
        self.assertAlmostEqual(
            np.mean(np.sum(cloud[:, 9:] ** 2, axis=1)), 0.5
        )

    def test_desynchronization_preserves_both_marginal_clouds(self):
        acceleration = self.signals["body_acc"][0]
        gyroscope = self.signals["body_gyro"][0]
        permutation = np.random.default_rng(2956).permutation(98)
        synchronized, _ = make_joint_point_cloud(
            acceleration, gyroscope, m=3, lag=15
        )
        desynchronized, _ = make_joint_point_cloud(
            acceleration,
            gyroscope,
            m=3,
            lag=15,
            gyroscope_permutation=permutation,
        )
        np.testing.assert_allclose(desynchronized[:, :9], synchronized[:, :9])
        np.testing.assert_allclose(
            desynchronized[:, 9:], synchronized[permutation, 9:]
        )
        self.assertFalse(np.allclose(desynchronized, synchronized))

    def test_equal_weight_coupled_metric_is_marginal_maximum(self):
        acceleration = self.signals["body_acc"][0]
        gyroscope = self.signals["body_gyro"][0]
        cloud, _ = make_joint_point_cloud(
            acceleration, gyroscope, m=3, lag=15
        )
        coupled, _ = make_coupled_distance_matrix(
            acceleration, gyroscope, m=3, lag=15
        )
        acceleration_distances = np.linalg.norm(
            cloud[:, None, :9] - cloud[None, :, :9], axis=2
        ) * np.sqrt(2.0)
        gyroscope_distances = np.linalg.norm(
            cloud[:, None, 9:] - cloud[None, :, 9:], axis=2
        ) * np.sqrt(2.0)
        np.testing.assert_allclose(
            coupled, np.maximum(acceleration_distances, gyroscope_distances)
        )

    def test_coupled_distance_is_a_metric(self):
        distances, _ = make_coupled_distance_matrix(
            self.signals["body_acc"][0],
            self.signals["body_gyro"][0],
            m=3,
            lag=15,
            acceleration_weight=0.25,
        )
        np.testing.assert_allclose(distances, distances.T)
        np.testing.assert_allclose(np.diag(distances), 0.0)
        indices = np.random.default_rng(2956).integers(0, len(distances), size=(100, 3))
        for i, j, k in indices:
            self.assertLessEqual(distances[i, k], distances[i, j] + distances[j, k] + 1e-12)

    def test_stable_rank_feature_shape(self):
        curves, scales = stable_rank_features(
            self.signals["body_acc"][:2], m=3, lag=15, normalize=True
        )
        self.assertEqual(curves.shape, (2, 2 * len(STABLE_RANK_GRID)))
        self.assertEqual(scales.shape, (2, 1))
        self.assertTrue(np.isfinite(curves).all())

    def test_joint_stable_rank_feature_shape(self):
        signals = {name: values[:2] for name, values in self.signals.items()}
        curves, scales = joint_sensor_stable_ranks(signals, m=3, lag=15)
        self.assertEqual(curves.shape, (2, 2 * len(STABLE_RANK_GRID)))
        self.assertEqual(scales.shape, (2, 2))
        self.assertTrue(np.isfinite(curves).all())

    def test_desynchronized_joint_feature_shape(self):
        signals = {name: values[:2] for name, values in self.signals.items()}
        curves, scales = joint_sensor_stable_ranks(
            signals, m=3, lag=15, desynchronize=True
        )
        self.assertEqual(curves.shape, (2, 2 * len(STABLE_RANK_GRID)))
        self.assertEqual(scales.shape, (2, 2))
        self.assertTrue(np.isfinite(curves).all())

    def test_coupled_stable_rank_feature_shape(self):
        signals = {name: values[:2] for name, values in self.signals.items()}
        curves, scales = coupled_sensor_stable_ranks(signals, m=3, lag=15)
        self.assertEqual(curves.shape, (2, 2 * len(STABLE_RANK_GRID)))
        self.assertEqual(scales.shape, (2, 2))
        self.assertTrue(np.isfinite(curves).all())

    def test_topological_dynamics_feature_shape(self):
        signals = {name: values[:2] for name, values in self.signals.items()}
        features, scales = multi_sensor_topological_dynamics(
            signals, m=3, lag=15
        )
        self.assertEqual(features.shape, (2, 16))
        self.assertEqual(scales.shape, (2, 2))
        self.assertTrue(np.isfinite(features).all())

    def test_circular_phase_recovers_known_frequency(self):
        time = np.arange(100) / 50.0
        angle = 2.0 * np.pi * time
        circle = np.column_stack((np.cos(angle), np.sin(angle)))
        result = ripser(circle, maxdim=1, do_cocycles=True, coeff=47)
        frequency, linearity = circular_phase_features(
            circle,
            result["dgms"][1],
            result["cocycles"][1],
        )
        self.assertAlmostEqual(frequency, 1.0, places=2)
        self.assertGreater(linearity, 0.99)

    def test_stable_rank_block_layout(self):
        features = np.arange(3 * 256).reshape(3, 256)
        blocks = stable_rank_blocks(features)
        self.assertEqual(blocks["Acceleration H0"].shape, (3, 64))
        self.assertEqual(blocks["Both sensors H1"].shape, (3, 128))
        np.testing.assert_array_equal(blocks["All H0+H1"], features)

    def test_stable_rank_distance_uses_grid_interval(self):
        zeros = np.zeros(len(STABLE_RANK_GRID))
        ones = np.ones(len(STABLE_RANK_GRID))
        self.assertAlmostEqual(stable_rank_l1(zeros, ones), 2.0)

    def test_subject_activity_prototypes(self):
        labels = np.tile(np.arange(1, 7), 2)
        subjects = np.repeat((1, 2), 6)
        features = np.column_stack((labels, subjects)).astype(float)
        prototypes, activities, groups = subject_activity_prototypes(
            features, labels, subjects
        )
        self.assertEqual(prototypes.shape, (12, 2))
        np.testing.assert_array_equal(activities, labels)
        np.testing.assert_array_equal(groups, subjects)

    def test_exact_sign_flip_probability(self):
        labels = np.tile(np.arange(1, 7), 3)
        subjects = np.repeat((1, 2, 3), 6)
        first = labels.copy()
        second = np.roll(labels.reshape(3, 6), 1, axis=1).reshape(-1)
        result = _paired_subject_difference(
            first, second, labels, subjects, repeats=100
        )
        self.assertEqual(result["exact_two_sided_sign_flip_p"], 0.25)


if __name__ == "__main__":
    unittest.main()
