"""Focused tests for the mathematical rewrite."""

import unittest

import numpy as np
from ripser import ripser

from data import first_subject_activity_windows, load_split
from evaluate import activity_distance_matrix
from tda import (
    GRID,
    _phase_features,
    delay_cloud,
    feature_matrices,
    normalize_cloud,
    stable_rank,
    stable_rank_distance,
)


class MathematicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sensors, labels, subjects = load_split("train")
        selected = first_subject_activity_windows(labels, subjects)[:12]
        cls.sensors = {name: values[selected] for name, values in sensors.items()}
        cls.labels = labels[selected]
        cls.subjects = subjects[selected]

    def test_delay_cloud_definition(self):
        path = np.arange(24).reshape(8, 3)
        cloud = delay_cloud(path, m=3, lag=2)
        self.assertEqual(cloud.shape, (4, 9))
        np.testing.assert_array_equal(
            cloud[0], np.concatenate((path[0], path[2], path[4]))
        )

    def test_normalization_removes_translation_and_scale(self):
        rng = np.random.default_rng(2956)
        cloud = rng.normal(size=(20, 4))
        first, _ = normalize_cloud(cloud)
        second, _ = normalize_cloud(7 * cloud + 3)
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(first.mean(axis=0), 0, atol=1e-12)
        self.assertAlmostEqual(np.sqrt(np.mean(np.sum(first**2, axis=1))), 1)

    def test_stable_rank_counts_lifetimes(self):
        diagram = np.asarray(((0.0, 0.5), (0.2, 1.2), (0.0, np.inf)))
        values = stable_rank(diagram, np.asarray((0.0, 0.75, 1.25)))
        np.testing.assert_array_equal(values, (2, 1, 0))

    def test_blockwise_stable_rank_distance(self):
        zeros = np.zeros(2 * len(GRID))
        ones = np.ones(2 * len(GRID))
        self.assertAlmostEqual(stable_rank_distance(zeros, ones), 4.0)

    def test_time_shuffle_is_deterministic_and_changes_delay_topology(self):
        original, _, _ = feature_matrices(self.sensors, m=3, lag=15)
        first, _, _ = feature_matrices(
            self.sensors, m=3, lag=15, shuffle_seed=2956
        )
        second, _, _ = feature_matrices(
            self.sensors, m=3, lag=15, shuffle_seed=2956
        )
        np.testing.assert_array_equal(first, second)
        self.assertGreater(stable_rank_distance(original, first).mean(), 0)

    def test_circular_phase_recovers_one_hertz(self):
        time = np.arange(100) / 50
        circle = np.column_stack((np.cos(2 * np.pi * time), np.sin(2 * np.pi * time)))
        result = ripser(circle, maxdim=1, do_cocycles=True, coeff=47)
        frequency, linearity = _phase_features(
            circle, result["dgms"][1], result["cocycles"][1], 50.0, 47
        )
        self.assertAlmostEqual(frequency, 1.0, places=2)
        self.assertGreater(linearity, 0.99)

    def test_real_feature_shapes_and_finiteness(self):
        ranks, scales, dynamics = feature_matrices(
            self.sensors, m=3, lag=15, include_dynamics=True
        )
        self.assertEqual(ranks.shape, (12, 256))
        self.assertEqual(scales.shape, (12, 2))
        self.assertEqual(dynamics.shape, (12, 16))
        self.assertTrue(np.isfinite(ranks).all())
        self.assertTrue(np.isfinite(scales).all())
        self.assertTrue(np.isfinite(dynamics).all())

    def test_activity_distance_matrix_shape(self):
        labels = np.tile(np.arange(1, 7), 2)
        subjects = np.repeat((1, 2), 6)
        features = np.repeat(np.eye(6), 2, axis=0).reshape(12, 6)
        features = np.pad(features, ((0, 0), (0, 58)))
        matrix = activity_distance_matrix(features, labels, subjects)
        self.assertEqual(matrix.shape, (6, 6))

    def test_official_subjects_are_disjoint(self):
        _, _, train_subjects = load_split("train")
        _, _, test_subjects = load_split("test")
        self.assertTrue(set(train_subjects).isdisjoint(test_subjects))


if __name__ == "__main__":
    unittest.main()
