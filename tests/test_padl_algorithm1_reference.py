import math
import unittest

import numpy as np
import torch

try:
    from padl_algorithm1_reference import (
        concentration_u,
        delta_stationarity_residual,
        fit_beta1_mle,
        normalize_columns,
        objective_components,
        rowwise_linf,
        solve_optimal_delta,
    )
except ImportError:
    from work.new_runner.padl_algorithm1_reference import (
        concentration_u,
        delta_stationarity_residual,
        fit_beta1_mle,
        normalize_columns,
        objective_components,
        rowwise_linf,
        solve_optimal_delta,
    )


class PADLAlgorithm1ReferenceTests(unittest.TestCase):
    def test_rowwise_linf_uses_atom_rows(self):
        R = torch.tensor(
            [
                [0.1, 0.9, 0.2],
                [0.8, 0.2, 0.6],
            ],
            dtype=torch.float64,
        )
        self.assertAlmostEqual(float(rowwise_linf(R).item()), 1.7)
        columnwise_value = torch.max(R, dim=0).values.sum().item()
        self.assertAlmostEqual(columnwise_value, 2.3)
        self.assertNotAlmostEqual(float(rowwise_linf(R).item()), columnwise_value)

    def test_objective_matches_direct_formula(self):
        X = torch.eye(2, dtype=torch.float64)
        D = torch.eye(2, dtype=torch.float64)
        R = torch.tensor([[0.5, 0.0], [0.0, 0.25]], dtype=torch.float64)
        report = objective_components(X, D, R, lambda1=2.0, lambda2=3.0)
        direct = (
            torch.square(X - D.matmul(R)).sum()
            + 2.0 * torch.abs(R).sum()
            + 3.0 * torch.max(torch.abs(R), dim=1).values.sum()
        )
        self.assertAlmostEqual(float(report["total"].item()), float(direct.item()))

    def test_beta1_mle_matches_closed_form(self):
        R = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)
        expected = R.numel() / float((-torch.log1p(-R)).sum().item())
        self.assertAlmostEqual(fit_beta1_mle(R), expected)

    def test_concentration_u_uses_plus_sign(self):
        value = concentration_u(200, beta=6.0, tau=0.05)
        expected = 200 / 7.0 + math.sqrt(100.0 * math.log(20.0))
        self.assertAlmostEqual(value, expected)

    def test_solved_delta_satisfies_stationarity(self):
        delta = solve_optimal_delta(
            N=200,
            beta=6.0,
            lambda1=5.0,
            lambda2=1.0,
            tau=0.05,
        )
        residual = delta_stationarity_residual(
            delta,
            N=200,
            beta=6.0,
            lambda1=5.0,
            lambda2=1.0,
            tau=0.05,
        )
        self.assertGreater(delta, 0.0)
        self.assertLess(delta, 1.0)
        self.assertLess(abs(residual), 1e-7)

    def test_normalize_columns(self):
        X = torch.tensor(
            [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=torch.float64
        )
        normalized, mean, std = normalize_columns(X)
        self.assertTrue(
            torch.allclose(
                normalized.mean(dim=0),
                torch.zeros(2, dtype=normalized.dtype),
            )
        )
        self.assertTrue(
            torch.allclose(
                normalized.std(dim=0, unbiased=False),
                torch.ones(2, dtype=normalized.dtype),
            )
        )
        self.assertTrue(torch.allclose(normalized * std + mean, X))

    def test_normalize_columns_maps_near_constant_float32_column_to_zero(self):
        X = torch.full((64, 1), 0.5019608, dtype=torch.float32)
        X[0, 0] = torch.nextafter(
            X[0, 0],
            torch.tensor(float("inf"), dtype=torch.float32),
        )
        normalized, mean, std = normalize_columns(X)
        self.assertTrue(torch.equal(normalized, torch.zeros_like(normalized)))
        self.assertTrue(torch.allclose(normalized * std + mean, mean.expand_as(X)))


if __name__ == "__main__":
    unittest.main()
