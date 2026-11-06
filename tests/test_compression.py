import unittest

import numpy as np

from src.compression import (
    CompreSSM,
    CompressionConfig,
    DiscreteSSM,
    compress_numpy,
)


class CompressionTests(unittest.TestCase):
    def test_rank_selection_retains_requested_hsv_sum(self):
        sigma = np.array([7.0, 2.0, 1.0])

        self.assertEqual(CompreSSM.rank_selection(sigma, threshold=0.8), 2)
        self.assertEqual(CompreSSM.rank_selection(sigma, threshold=1.0), 3)
        self.assertEqual(CompreSSM.rank_selection(sigma, threshold=0.0), 1)

    def test_transform_does_not_floor_small_positive_hsvs(self):
        _, diag_entries = CompreSSM.calculate_transform_matrix(
            np.eye(1),
            r=1,
            sigma=np.array([1e-16]),
            Vh=np.eye(1),
        )

        np.testing.assert_allclose(diag_entries, [1e8])

    def test_balanced_truncation_projects_to_top_hsv(self):
        system = DiscreteSSM.from_numpy(
            A=np.diag([0.5, 0.9]),
            B=np.eye(2),
            C=np.eye(2),
            D=np.zeros((2, 2)),
        )

        reduced = compress_numpy(
            system,
            config=CompressionConfig(tol=0.25, max_ratio=0.5),
        )

        self.assertIsNotNone(reduced)
        self.assertEqual(reduced.r, 1)
        np.testing.assert_allclose(reduced.A, [[0.9]], atol=1e-12)
        np.testing.assert_allclose(reduced.D, system.D)
        self.assertAlmostEqual(reduced.error_bound, 2 / (1 - 0.5**2))

    def test_noop_when_rank_is_not_reduced(self):
        system = DiscreteSSM.from_numpy(
            A=np.diag([0.5, 0.9]),
            B=np.eye(2),
            C=np.eye(2),
            D=np.zeros((2, 2)),
        )

        reduced = compress_numpy(
            system,
            config=CompressionConfig(tol=0.0, max_ratio=1.0),
        )

        self.assertIsNone(reduced)

    def test_continuous_discretization_is_stable_for_small_rates(self):
        system = DiscreteSSM.from_continuous_diagonal_numpy(
            a=np.array([-1e-12]),
            B=np.array([[2.0]]),
            C=np.array([[3.0]]),
            D=np.zeros((1, 1)),
            dt=0.1,
        )

        np.testing.assert_allclose(system.A, [[np.exp(-1e-13)]])
        np.testing.assert_allclose(system.B, [[0.2]], rtol=1e-12)

    def test_rejects_incompatible_state_space_dimensions(self):
        with self.assertRaisesRegex(ValueError, "same number of rows"):
            DiscreteSSM.from_numpy(
                A=np.eye(2),
                B=np.ones((1, 1)),
                C=np.eye(2),
                D=np.zeros((2, 1)),
            )


if __name__ == "__main__":
    unittest.main()
