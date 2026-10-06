import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parents[1] / "rebuttal_tools"
import sys

sys.path.insert(0, str(TOOLS_DIR))

from artifact_io import save_active_span_artifact, save_warmup_artifact
from paired_run_statistics import analyze, exact_sign_flip_pvalue, holm_adjust
from run_metrics_io import append_run_row, read_run_rows, run_exists, tensor_fingerprint
from theory_assumption_diagnostics import (
    compute_activation_conditioned_taylor_diagnostics,
    compute_active_span_diagnostics,
    compute_beta_activation_conditioned_taylor_expectations,
    compute_taylor_diagnostics,
    compute_taylor_regime_diagnostics,
    load_and_compute,
)


class TheoryDiagnosticsTest(unittest.TestCase):
    def test_taylor_terms_match_direct_computation(self):
        R = np.asarray([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64)
        beta = 3.0
        report = compute_taylor_diagnostics(R, beta)
        self.assertAlmostEqual(report["t1_linear"], 2.0 * R.sum())
        self.assertAlmostEqual(report["t2_quadratic"], np.square(R).sum())
        self.assertGreater(report["exact_remainder"], report["t2_quadratic"])
        self.assertGreater(report["t2_over_exact_remainder"], 0.0)
        self.assertLess(report["t2_over_exact_remainder"], 1.0)

    def test_taylor_remainder_obeys_finite_geometric_bounds(self):
        R = np.asarray([[0.1, 0.2], [0.05, 0.15]], dtype=np.float64)
        report = compute_taylor_diagnostics(R, beta_hat=3.0)
        self.assertAlmostEqual(report["max_coefficient"], 0.2)
        self.assertAlmostEqual(report["fitted_prior_mean"], 0.25)
        self.assertAlmostEqual(
            report["exact_remainder_lower_bound"],
            report["t2_quadratic"],
        )
        self.assertAlmostEqual(
            report["exact_remainder_upper_bound"],
            report["t2_quadratic"] / (1.0 - 0.2),
        )
        self.assertLessEqual(
            report["exact_remainder_lower_bound"],
            report["exact_remainder"],
        )
        self.assertLessEqual(
            report["exact_remainder"],
            report["exact_remainder_upper_bound"],
        )
        pointwise = -np.log1p(-R) - R
        expected = np.percentile(pointwise.reshape(-1), (0, 50, 90, 95, 99, 100))
        for percentile, value in zip((0, 50, 90, 95, 99, 100), expected):
            self.assertAlmostEqual(
                report["pointwise_remainder_percentiles"][str(percentile)],
                float(value),
            )

    def test_taylor_regimes_expose_large_coefficient_tail_contribution(self):
        R = np.asarray([[0.0, 0.1, 0.2, 0.4, 0.8]], dtype=np.float64)
        report = compute_taylor_regime_diagnostics(
            R,
            beta_hat=3.0,
            cutoffs=(0.1, 0.2, 0.3, 0.5),
        )
        self.assertEqual(
            [item["count"] for item in report["regimes"]],
            [2, 1, 0, 1, 1],
        )
        self.assertAlmostEqual(report["overall"]["linear_energy"], 3.0)
        self.assertAlmostEqual(
            report["overall"]["exact_remainder"],
            1.897535206344254,
        )
        tail = report["regimes"][-1]
        self.assertEqual(tail["label"], "(0.5, 1)")
        self.assertAlmostEqual(
            tail["exact_remainder_over_linear"],
            1.0117973905426254,
        )
        self.assertAlmostEqual(
            tail["fraction_of_total_exact_remainder"],
            0.8531466607078602,
        )

    def test_activation_conditioned_taylor_uses_strict_rowwise_threshold(self):
        R = np.asarray(
            [
                [0.1, 0.2, 0.6],
                [0.05, 0.1, 0.5],
                [0.01, 0.02, 0.03],
            ],
            dtype=np.float64,
        )
        report = compute_activation_conditioned_taylor_diagnostics(
            R,
            beta_hat=3.0,
            delta=0.5,
        )
        self.assertAlmostEqual(
            report["inactive_row_deterministic_bounds"][
                "quadratic_over_linear_upper_bound"
            ],
            0.25,
        )
        self.assertAlmostEqual(
            report["inactive_row_deterministic_bounds"][
                "exact_remainder_over_linear_upper_bound"
            ],
            (-np.log(0.5) - 0.5) / 0.5,
        )
        self.assertAlmostEqual(
            report["active_row_deterministic_bounds"][
                "quadratic_over_linear_supremum"
            ],
            0.5,
        )
        self.assertIsNone(
            report["active_row_deterministic_bounds"][
                "exact_remainder_over_linear_upper_bound"
            ]
        )
        self.assertEqual(report["active_row_count"], 1)
        self.assertEqual(report["inactive_row_count"], 2)
        self.assertEqual(report["active_rows"]["coefficient_count"], 3)
        self.assertAlmostEqual(
            report["active_rows"]["linear_energy"],
            2.0 * (0.1 + 0.2 + 0.6),
        )
        self.assertAlmostEqual(
            report["active_rows"]["quadratic_energy"],
            0.1 ** 2 + 0.2 ** 2 + 0.6 ** 2,
        )
        expected_remainder = 2.0 * sum(
            -np.log1p(-value) - value
            for value in (0.1, 0.2, 0.6)
        )
        self.assertAlmostEqual(
            report["active_rows"]["exact_remainder"],
            expected_remainder,
        )
        active = report["active_rows"]
        expected_higher_order = (
            expected_remainder - active["quadratic_energy"]
        )
        self.assertAlmostEqual(
            active["second_order_remainder"],
            expected_higher_order,
        )
        self.assertAlmostEqual(
            active["quadratic_fraction_of_exact_remainder"],
            active["quadratic_energy"] / expected_remainder,
        )
        self.assertAlmostEqual(
            active["first_order_remainder_fraction_of_exact_prior"],
            expected_remainder
            / (active["linear_energy"] + expected_remainder),
        )
        self.assertAlmostEqual(
            active["second_order_remainder_fraction_of_exact_prior"],
            expected_higher_order
            / (active["linear_energy"] + expected_remainder),
        )
        self.assertAlmostEqual(
            active["second_order_remainder_over_second_order_approx"],
            expected_higher_order
            / (active["linear_energy"] + active["quadratic_energy"]),
        )
        self.assertAlmostEqual(
            report["active_rows"]["fraction_of_total_linear"]
            + report["inactive_rows"]["fraction_of_total_linear"],
            1.0,
        )
        self.assertAlmostEqual(
            report["active_rows"]["fraction_of_total_exact_remainder"]
            + report["inactive_rows"]["fraction_of_total_exact_remainder"],
            1.0,
        )

    def test_beta_activation_conditioning_matches_hand_integrated_case(self):
        report = compute_beta_activation_conditioned_taylor_expectations(
            beta=2.0,
            delta=0.5,
            samples_per_row=1,
            nominal_atoms=128,
        )
        self.assertAlmostEqual(report["cdf_at_delta"], 0.75)
        self.assertAlmostEqual(report["probability_row_inactive"], 0.75)
        self.assertAlmostEqual(report["probability_row_active"], 0.25)
        self.assertAlmostEqual(
            report["log_probability_row_inactive"],
            np.log(0.75),
        )
        self.assertAlmostEqual(report["expected_active_rows"], 32.0)
        self.assertAlmostEqual(report["expected_inactive_rows"], 96.0)
        self.assertAlmostEqual(
            report["inactive_row_deterministic_bounds"][
                "quadratic_over_linear_upper_bound"
            ],
            0.25,
        )
        self.assertAlmostEqual(
            report["inactive_row_deterministic_bounds"][
                "exact_remainder_over_linear_upper_bound"
            ],
            (-np.log(0.5) - 0.5) / 0.5,
        )
        self.assertAlmostEqual(
            report["active_row_deterministic_bounds"][
                "quadratic_over_linear_supremum"
            ],
            0.5,
        )
        self.assertIsNone(
            report["active_row_deterministic_bounds"][
                "exact_remainder_over_linear_upper_bound"
            ]
        )
        self.assertAlmostEqual(
            report["unconditional"]["quadratic_over_linear"],
            0.25,
        )
        self.assertAlmostEqual(
            report["unconditional"]["exact_remainder_over_linear"],
            0.5,
        )
        self.assertAlmostEqual(
            report["inactive_rows"]["quadratic_over_linear"],
            0.15625,
        )
        self.assertAlmostEqual(
            report["inactive_rows"]["exact_remainder_over_linear"],
            0.21027922916008213,
        )
        self.assertAlmostEqual(
            report["active_rows"]["quadratic_over_linear"],
            0.34375,
        )
        self.assertAlmostEqual(
            report["active_rows"]["exact_remainder_over_linear"],
            0.7897207708399179,
        )
        self.assertAlmostEqual(
            report["active_rows"]["second_order_remainder_over_linear"],
            0.7897207708399179 - 0.34375,
        )
        self.assertAlmostEqual(
            report["active_rows"]["quadratic_fraction_of_exact_remainder"],
            0.34375 / 0.7897207708399179,
        )
        self.assertAlmostEqual(
            report["active_rows"][
                "first_order_remainder_fraction_of_exact_prior"
            ],
            0.7897207708399179 / (1.0 + 0.7897207708399179),
        )
        self.assertAlmostEqual(
            report["active_rows"][
                "second_order_remainder_fraction_of_exact_prior"
            ],
            (0.7897207708399179 - 0.34375)
            / (1.0 + 0.7897207708399179),
        )
        self.assertAlmostEqual(
            report["active_rows"][
                "fraction_of_unconditional_linear_energy"
            ],
            0.5,
        )
        self.assertAlmostEqual(
            report["active_rows"][
                "fraction_of_unconditional_quadratic_energy"
            ],
            0.6875,
        )
        self.assertAlmostEqual(
            report["active_rows"][
                "fraction_of_unconditional_exact_remainder"
            ],
            0.7897207708399179,
        )

    def test_beta_activation_conditioning_keeps_rare_active_tail_finite(self):
        delta = 1.0 - 1e-9
        report = compute_beta_activation_conditioned_taylor_expectations(
            beta=2.0,
            delta=delta,
            samples_per_row=1,
        )
        self.assertAlmostEqual(
            report["probability_row_active"],
            (1.0 - delta) ** 2,
            delta=1e-30,
        )
        self.assertAlmostEqual(
            report["active_rows"]["mean_r"],
            1.0 - 2.0 * (1.0 - delta) / 3.0,
            places=14,
        )
        self.assertIsNotNone(
            report["active_rows"]["exact_remainder_over_linear"]
        )

    def test_beta_activation_conditioning_is_stable_for_small_delta(self):
        delta = 1e-6
        report = compute_beta_activation_conditioned_taylor_expectations(
            beta=2.0,
            delta=delta,
            samples_per_row=200,
        )
        low_r_mass = delta ** 2 - 2.0 * delta ** 3 / 3.0
        low_r2_mass = 2.0 * delta ** 3 / 3.0 - delta ** 4 / 2.0
        low_remainder_mass = 2.0 * sum(
            delta ** (n + 1) / (n * (n + 1))
            - delta ** (n + 2) / (n * (n + 2))
            for n in range(2, 9)
        )
        self.assertAlmostEqual(
            report["inactive_rows"]["quadratic_over_linear"],
            0.5 * low_r2_mass / low_r_mass,
            delta=1e-15,
        )
        self.assertAlmostEqual(
            report["inactive_rows"]["exact_remainder_over_linear"],
            low_remainder_mass / low_r_mass,
            delta=1e-15,
        )
        self.assertGreater(
            report["inactive_rows"]["exact_remainder_over_linear"],
            0.0,
        )

    def test_active_span_identity_dictionary_has_alpha_one(self):
        D = np.eye(2, dtype=np.float64)
        R_warmup = np.asarray([[0.8, 0.1], [0.1, 0.8]], dtype=np.float64)
        R_effective = np.zeros((2, 2), dtype=np.float64)
        X_norm = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
        report = compute_active_span_diagnostics(
            D, X_norm, R_warmup, R_effective, delta=0.5
        )
        self.assertEqual(report["active_count"], 2)
        self.assertAlmostEqual(report["alpha_empirical_min"], 1.0)
        self.assertAlmostEqual(
            report["geometry_factor_active_count_over_alpha"], 2.0
        )
        self.assertAlmostEqual(report["dictionary_column_norm_max_abs_error"], 0.0)
        self.assertAlmostEqual(report["x_column_mean_max_abs"], 1.0)

    def test_active_set_is_strict_and_residual_uses_effective_coefficients(self):
        D = np.eye(2, dtype=np.float64)
        X_norm = np.asarray([[1.0], [-1.0]], dtype=np.float64)
        R_activation = np.asarray([[0.5], [0.6]], dtype=np.float64)
        R_effective = np.asarray([[1.0], [0.0]], dtype=np.float64)
        report = compute_active_span_diagnostics(
            D,
            X_norm,
            R_activation,
            R_effective,
            delta=0.5,
        )
        self.assertEqual(report["active_count"], 1)
        self.assertAlmostEqual(report["alpha_empirical_min"], 1.0)

    def test_empty_active_set_is_reported_as_unevaluable(self):
        report = compute_active_span_diagnostics(
            np.eye(2, dtype=np.float64),
            np.asarray([[1.0], [-1.0]], dtype=np.float64),
            np.asarray([[0.1], [0.2]], dtype=np.float64),
            np.zeros((2, 1), dtype=np.float64),
            delta=0.5,
        )
        self.assertEqual(report["active_count"], 0)
        self.assertFalse(report["condition_evaluable"])
        self.assertEqual(report["unevaluable_reason"], "empty_active_set")
        self.assertIsNone(report["alpha_empirical_min"])

    def test_artifact_requires_formal_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact.npz"
            np.savez_compressed(
                artifact,
                stage=np.asarray("test_hard_gate"),
                R_warmup=np.asarray([[0.2]], dtype=np.float64),
                beta_hat=np.asarray(2.0),
            )
            with self.assertRaises(ValueError):
                load_and_compute(str(artifact))

    def test_saved_artifact_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact.npz"
            save_warmup_artifact(
                str(artifact),
                R_warmup=np.asarray([[0.01, 0.02]], dtype=np.float64),
                beta_hat=2.5,
                config={"dataset": "synthetic"},
            )
            report = load_and_compute(str(artifact))
            self.assertEqual(report["artifact_stage"], "warmup_pre_gate")
            self.assertIsNone(report["active_span"])
            self.assertEqual(
                [item["label"] for item in report["taylor_regimes"]["regimes"]],
                [
                    "[0, 0.1]",
                    "(0.1, 0.2]",
                    "(0.2, 0.3]",
                    "(0.3, 0.5]",
                    "(0.5, 1)",
                ],
            )

    def test_separate_active_span_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            warmup = Path(tmp) / "warmup.npz"
            active = Path(tmp) / "active.npz"
            provenance = {
                "dataset": "synthetic",
                "seed": 7,
                "split_id": "fixed",
                "data_fingerprint": "abc",
            }
            save_warmup_artifact(
                str(warmup),
                R_warmup=np.asarray([[0.01, 0.02, 0.03]], dtype=np.float64),
                beta_hat=2.5,
                config=provenance,
            )
            save_active_span_artifact(
                str(active),
                D=np.eye(2, dtype=np.float64),
                X_norm=np.asarray(
                    [[1.0, 0.0], [0.0, 2.0]], dtype=np.float64
                ),
                R_activation=np.asarray(
                    [[0.8, 0.1], [0.1, 0.8]], dtype=np.float64
                ),
                R_effective=np.zeros((2, 2), dtype=np.float64),
                delta=0.5,
                config=provenance,
            )
            report = load_and_compute(
                str(warmup), active_span_artifact_path=str(active)
            )
            self.assertEqual(report["active_span"]["active_count"], 2)
            self.assertAlmostEqual(
                report["active_span"]["alpha_empirical_min"], 1.0
            )

    def test_separate_artifacts_reject_mismatched_data_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            warmup = Path(tmp) / "warmup.npz"
            active = Path(tmp) / "active.npz"
            base_config = {
                "dataset": "synthetic",
                "seed": 7,
                "split_id": "fixed",
            }
            save_warmup_artifact(
                str(warmup),
                R_warmup=np.asarray([[0.01, 0.02, 0.03]], dtype=np.float64),
                beta_hat=2.5,
                config=dict(base_config, data_fingerprint="warmup-data"),
            )
            save_active_span_artifact(
                str(active),
                D=np.eye(2, dtype=np.float64),
                X_norm=np.asarray(
                    [[1.0, 0.0], [0.0, 2.0]], dtype=np.float64
                ),
                R_activation=np.asarray(
                    [[0.8, 0.1], [0.1, 0.8]], dtype=np.float64
                ),
                R_effective=np.zeros((2, 2), dtype=np.float64),
                delta=0.5,
                config=dict(base_config, data_fingerprint="evaluation-data"),
            )
            with self.assertRaisesRegex(ValueError, "data_fingerprint"):
                load_and_compute(
                    str(warmup), active_span_artifact_path=str(active)
                )


class StatisticsTest(unittest.TestCase):
    def test_exact_sign_flip(self):
        p_value = exact_sign_flip_pvalue([1.0, 1.0])
        self.assertAlmostEqual(p_value, 0.5)

    def test_holm_is_monotone_in_sorted_order(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(len(adjusted), 3)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))
        self.assertGreaterEqual(adjusted[0], 0.01)

    def test_paired_analysis_matches_by_run_keys(self):
        rows = []
        for seed in range(1, 6):
            common = {
                "dataset": "synthetic",
                "seed": str(seed),
                "split_id": "fixed",
                "data_fingerprint": "abc",
            }
            rows.append(dict(common, method="padl", rmse=str(0.1 + seed * 0.001)))
            rows.append(dict(common, method="baseline", rmse=str(0.2 + seed * 0.001)))
        results = analyze(
            rows,
            reference="padl",
            comparators=["baseline"],
            metrics=["rmse"],
            unpaired=False,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["n_pairs"], 5)
        self.assertAlmostEqual(
            results[0]["mean_difference_reference_minus_comparator"], -0.1
        )

    def test_append_only_run_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            row = {
                "dataset": "synthetic",
                "seed": 1,
                "split_id": "fixed",
                "method": "padl",
                "rmse": 0.1,
                "data_fingerprint": tensor_fingerprint(np.arange(4)),
            }
            append_run_row(str(path), row)
            self.assertTrue(run_exists(str(path), row))
            self.assertEqual(len(read_run_rows(str(path))), 1)
            with self.assertRaises(ValueError):
                append_run_row(str(path), row)


if __name__ == "__main__":
    unittest.main()
