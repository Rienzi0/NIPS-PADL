#!/usr/bin/env python3
"""Compute PADL Taylor and restricted active-span diagnostics from NPZ files.

Taylor analysis requires ``stage == "warmup_pre_gate"``. Optional Eq. 19
analysis uses a separate ``stage == "evaluation_coefficients"`` artifact so
that warm-up coefficients are not confused with final evaluation codes.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


REQUIRED_STAGE = "warmup_pre_gate"
ACTIVE_SPAN_STAGE = "evaluation_coefficients"
SMALL_THRESHOLDS = (0.1, 0.2, 0.3)
PERCENTILES = (0, 1, 5, 25, 50, 75, 90, 95, 99, 100)
REMAINDER_PERCENTILES = (0, 50, 90, 95, 99, 100)
MATCHED_PROVENANCE_FIELDS = (
    "dataset",
    "seed",
    "split_id",
    "data_fingerprint",
)


def _as_scalar_string(value: np.ndarray) -> str:
    scalar = np.asarray(value)
    if scalar.size != 1:
        raise ValueError("Expected a scalar string metadata field.")
    return str(scalar.reshape(()).item())


def _as_scalar_float(value: np.ndarray, name: str) -> float:
    scalar = np.asarray(value)
    if scalar.size != 1:
        raise ValueError("{} must be scalar.".format(name))
    result = float(scalar.reshape(()).item())
    if not np.isfinite(result):
        raise ValueError("{} must be finite.".format(name))
    return result


def _as_config(value: np.ndarray, name: str) -> Dict[str, Any]:
    raw = _as_scalar_string(value)
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("{} must contain valid JSON.".format(name)) from error
    if not isinstance(config, dict):
        raise ValueError("{} must contain a JSON object.".format(name))
    return config


def _validate_matching_provenance(
    warmup_config: Dict[str, Any],
    active_config: Dict[str, Any],
) -> None:
    for field in MATCHED_PROVENANCE_FIELDS:
        if field not in warmup_config or field not in active_config:
            raise ValueError(
                "Both artifacts must record provenance field {!r}.".format(field)
            )
        if str(warmup_config[field]) != str(active_config[field]):
            raise ValueError(
                "Artifact provenance mismatch for {}: {!r} != {!r}.".format(
                    field,
                    warmup_config[field],
                    active_config[field],
                )
            )


def _validate_matrix(
    value: np.ndarray,
    name: str,
    *,
    ndim: int = 2,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError("{} must have {} dimensions; got {}.".format(name, ndim, array.ndim))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} contains NaN or infinity.".format(name))
    return array


def compute_taylor_diagnostics(R: np.ndarray, beta_hat: float) -> Dict[str, Any]:
    """Compute first-, second-, and exact Beta negative-log-prior terms."""

    R = _validate_matrix(R, "R_warmup")
    if np.any(R < 0.0) or np.any(R >= 1.0):
        raise ValueError("R_warmup must lie in [0, 1) before gating.")
    if beta_hat <= 1.0:
        raise ValueError("beta_hat must be > 1 for the stated sparsity calibration.")

    coefficient = beta_hat - 1.0
    t1 = coefficient * float(R.sum())
    t2 = 0.5 * coefficient * float(np.square(R).sum())
    t3 = (coefficient / 3.0) * float(np.power(R, 3).sum())
    exact = coefficient * float((-np.log1p(-R)).sum())
    pointwise_remainder = -np.log1p(-R) - R
    exact_remainder = exact - t1
    second_order_remainder = exact - (t1 + t2)

    flat = R.reshape(-1)
    max_coefficient = float(flat.max()) if flat.size else 0.0
    exact_remainder_lower_bound = t2
    exact_remainder_upper_bound = (
        float(t2 / (1.0 - max_coefficient))
        if max_coefficient < 1.0
        else float("inf")
    )
    percentile_values = np.percentile(flat, PERCENTILES)

    report: Dict[str, Any] = {
        "shape": [int(x) for x in R.shape],
        "num_coefficients": int(R.size),
        "beta_hat": float(beta_hat),
        "fitted_prior_mean": float(1.0 / (1.0 + beta_hat)),
        "mean": float(flat.mean()),
        "std": float(flat.std(ddof=0)),
        "max_coefficient": max_coefficient,
        "t1_linear": t1,
        "t2_quadratic": t2,
        "t3_cubic": t3,
        "t2_over_t1": float(t2 / t1) if t1 > 0 else None,
        "t3_over_t1": float(t3 / t1) if t1 > 0 else None,
        "exact_energy": exact,
        "exact_remainder": exact_remainder,
        "exact_remainder_over_t1": (
            float(exact_remainder / t1) if t1 > 0 else None
        ),
        "exact_remainder_over_exact": (
            float(exact_remainder / exact) if exact > 0 else None
        ),
        "exact_remainder_lower_bound": exact_remainder_lower_bound,
        "exact_remainder_upper_bound": exact_remainder_upper_bound,
        "exact_remainder_upper_bound_over_t1": (
            float(exact_remainder_upper_bound / t1) if t1 > 0 else None
        ),
        "t2_over_exact_remainder": (
            float(t2 / exact_remainder) if exact_remainder > 0 else None
        ),
        "second_order_remainder": second_order_remainder,
        "second_order_remainder_over_exact": (
            float(second_order_remainder / exact) if exact > 0 else None
        ),
        "percentiles": {
            str(p): float(value)
            for p, value in zip(PERCENTILES, percentile_values)
        },
        "fractions_at_or_below": {
            str(threshold): float(np.mean(flat <= threshold))
            for threshold in SMALL_THRESHOLDS
        },
        "pointwise_remainder_percentiles": {
            str(p): float(value)
            for p, value in zip(
                REMAINDER_PERCENTILES,
                np.percentile(
                    pointwise_remainder.reshape(-1),
                    REMAINDER_PERCENTILES,
                ),
            )
        },
    }
    return report


def compute_taylor_regime_diagnostics(
    R: np.ndarray,
    beta_hat: float,
    *,
    cutoffs: Iterable[float] = (0.1, 0.2, 0.3, 0.5),
) -> Dict[str, Any]:
    """Break Taylor energy into mutually exclusive coefficient regimes."""

    R = _validate_matrix(R, "R_warmup")
    if np.any(R < 0.0) or np.any(R >= 1.0):
        raise ValueError("R_warmup must lie in [0, 1) before gating.")
    if beta_hat <= 1.0 or not np.isfinite(beta_hat):
        raise ValueError("beta_hat must be finite and greater than one.")

    edges = tuple(float(value) for value in cutoffs)
    if (
        not edges
        or any(not np.isfinite(value) for value in edges)
        or any(value <= 0.0 or value >= 1.0 for value in edges)
        or any(left >= right for left, right in zip(edges, edges[1:]))
    ):
        raise ValueError("cutoffs must be strictly increasing values in (0, 1).")

    flat = R.reshape(-1)
    scale = float(beta_hat - 1.0)
    linear_values = scale * flat
    quadratic_values = 0.5 * scale * np.square(flat)
    remainder_values = scale * (-np.log1p(-flat) - flat)
    total_linear = float(linear_values.sum())
    total_quadratic = float(quadratic_values.sum())
    total_remainder = float(remainder_values.sum())

    masks = [flat <= edges[0]]
    for lower, upper in zip(edges, edges[1:]):
        masks.append((flat > lower) & (flat <= upper))
    masks.append(flat > edges[-1])

    labels = ["[0, {:g}]".format(edges[0])]
    labels.extend(
        "({:g}, {:g}]".format(lower, upper)
        for lower, upper in zip(edges, edges[1:])
    )
    labels.append("({:g}, 1)".format(edges[-1]))

    regimes = []
    for label, mask in zip(labels, masks):
        linear = float(linear_values[mask].sum())
        quadratic = float(quadratic_values[mask].sum())
        remainder = float(remainder_values[mask].sum())
        regimes.append(
            {
                "label": label,
                "count": int(mask.sum()),
                "fraction_of_coefficients": float(mask.mean()),
                "sum_r": float(flat[mask].sum()),
                "linear_energy": linear,
                "quadratic_energy": quadratic,
                "exact_remainder": remainder,
                "quadratic_over_linear": (
                    float(quadratic / linear) if linear > 0.0 else None
                ),
                "exact_remainder_over_linear": (
                    float(remainder / linear) if linear > 0.0 else None
                ),
                "fraction_of_total_linear": (
                    float(linear / total_linear) if total_linear > 0.0 else None
                ),
                "fraction_of_total_exact_remainder": (
                    float(remainder / total_remainder)
                    if total_remainder > 0.0
                    else None
                ),
            }
        )

    return {
        "cutoffs": list(edges),
        "overall": {
            "count": int(flat.size),
            "sum_r": float(flat.sum()),
            "linear_energy": total_linear,
            "quadratic_energy": total_quadratic,
            "exact_remainder": total_remainder,
            "quadratic_over_linear": (
                float(total_quadratic / total_linear)
                if total_linear > 0.0
                else None
            ),
            "exact_remainder_over_linear": (
                float(total_remainder / total_linear)
                if total_linear > 0.0
                else None
            ),
        },
        "regimes": regimes,
    }


def _summarize_taylor_rows(
    R: np.ndarray,
    beta_hat: float,
    row_mask: np.ndarray,
    *,
    total_linear: float,
    total_quadratic: float,
    total_remainder: float,
) -> Dict[str, Any]:
    selected = R[row_mask]
    flat = selected.reshape(-1)
    scale = float(beta_hat - 1.0)
    linear = scale * float(flat.sum())
    quadratic = 0.5 * scale * float(np.square(flat).sum())
    remainder = scale * float((-np.log1p(-flat) - flat).sum())
    second_order_remainder = remainder - quadratic
    exact_prior = linear + remainder
    second_order_approx = linear + quadratic
    return {
        "row_count": int(row_mask.sum()),
        "coefficient_count": int(flat.size),
        "mean_coefficient": float(flat.mean()) if flat.size else None,
        "linear_energy": linear,
        "quadratic_energy": quadratic,
        "exact_remainder": remainder,
        "second_order_remainder": second_order_remainder,
        "quadratic_over_linear": (
            float(quadratic / linear) if linear > 0.0 else None
        ),
        "exact_remainder_over_linear": (
            float(remainder / linear) if linear > 0.0 else None
        ),
        "second_order_remainder_over_linear": (
            float(second_order_remainder / linear)
            if linear > 0.0
            else None
        ),
        "quadratic_fraction_of_exact_remainder": (
            float(quadratic / remainder) if remainder > 0.0 else None
        ),
        "first_order_remainder_fraction_of_exact_prior": (
            float(remainder / exact_prior) if exact_prior > 0.0 else None
        ),
        "second_order_remainder_fraction_of_exact_prior": (
            float(second_order_remainder / exact_prior)
            if exact_prior > 0.0
            else None
        ),
        "second_order_remainder_over_second_order_approx": (
            float(second_order_remainder / second_order_approx)
            if second_order_approx > 0.0
            else None
        ),
        "fraction_of_total_linear": (
            float(linear / total_linear) if total_linear > 0.0 else None
        ),
        "fraction_of_total_quadratic": (
            float(quadratic / total_quadratic)
            if total_quadratic > 0.0
            else None
        ),
        "fraction_of_total_exact_remainder": (
            float(remainder / total_remainder)
            if total_remainder > 0.0
            else None
        ),
    }


def _inactive_row_taylor_bounds(delta: float) -> Dict[str, float]:
    if delta == 0.0:
        exact_upper = 0.0
    else:
        exact_upper = float(
            (-np.log1p(-delta) - delta) / delta
        )
    return {
        "quadratic_over_linear_upper_bound": float(delta / 2.0),
        "exact_remainder_over_linear_upper_bound": exact_upper,
    }


def _active_row_taylor_bounds() -> Dict[str, Any]:
    return {
        "quadratic_over_linear_supremum": 0.5,
        "exact_remainder_over_linear_upper_bound": None,
        "note": (
            "For nonzero rows with coefficients in [0, 1), quadratic / "
            "linear is strictly below 1/2. The active event supplies no "
            "finite exact-remainder / linear bound determined by delta "
            "alone."
        ),
    }


def _beta_low_partial_moments(
    beta: float,
    delta: float,
) -> tuple:
    """Evaluate truncated moments without small-delta cancellation."""

    log_one_minus_delta = np.log1p(-delta)
    log_tail_probability = beta * log_one_minus_delta
    if abs(log_tail_probability) < 0.1:
        nodes, weights = np.polynomial.legendre.leggauss(32)
        values = 0.5 * delta * (nodes + 1.0)
        remainder = np.zeros_like(values)
        for power in range(2, 16):
            remainder += np.power(values, power) / float(power)
        density = beta * np.power(1.0 - values, beta - 1.0)
        integration_scale = 0.5 * delta
        return (
            float(
                integration_scale
                * np.dot(weights, values * density)
            ),
            float(
                integration_scale
                * np.dot(weights, np.square(values) * density)
            ),
            float(
                integration_scale
                * np.dot(weights, remainder * density)
            ),
        )

    tail_probability = float(np.exp(log_tail_probability))
    cdf = float(-np.expm1(log_tail_probability))
    low_r_mass = float(
        -np.expm1(
            log_tail_probability + np.log1p(beta * delta)
        )
        / (beta + 1.0)
    )
    second_moment_factor = (
        beta * delta
        + 0.5 * beta * (beta + 1.0) * delta ** 2
    )
    low_r2_mass = float(
        -2.0
        * np.expm1(
            log_tail_probability + np.log1p(second_moment_factor)
        )
        / ((beta + 1.0) * (beta + 2.0))
    )
    low_log_mass = (
        cdf / beta
        + tail_probability * log_one_minus_delta
    )
    return low_r_mass, low_r2_mass, float(low_log_mass - low_r_mass)


def compute_activation_conditioned_taylor_diagnostics(
    R: np.ndarray,
    beta_hat: float,
    delta: float,
) -> Dict[str, Any]:
    """Split Taylor energies by the strict row-wise activation rule."""

    R = _validate_matrix(R, "R_warmup")
    if np.any(R < 0.0) or np.any(R >= 1.0):
        raise ValueError("R_warmup must lie in [0, 1) before gating.")
    if beta_hat <= 1.0 or not np.isfinite(beta_hat):
        raise ValueError("beta_hat must be finite and greater than one.")
    if not np.isfinite(delta) or delta < 0.0 or delta >= 1.0:
        raise ValueError("delta must lie in [0, 1).")

    scale = float(beta_hat - 1.0)
    total_linear = scale * float(R.sum())
    total_quadratic = 0.5 * scale * float(np.square(R).sum())
    total_remainder = scale * float((-np.log1p(-R) - R).sum())
    row_peaks = np.max(np.abs(R), axis=1)
    active_mask = row_peaks > float(delta)
    inactive_mask = ~active_mask
    all_mask = np.ones(R.shape[0], dtype=bool)

    return {
        "delta": float(delta),
        "activation_rule": "max_i_abs_R_ki_strictly_greater_than_delta",
        "total_row_count": int(R.shape[0]),
        "samples_per_row": int(R.shape[1]),
        "active_row_count": int(active_mask.sum()),
        "inactive_row_count": int(inactive_mask.sum()),
        "active_row_fraction": float(active_mask.mean()),
        "inactive_row_deterministic_bounds": (
            _inactive_row_taylor_bounds(delta)
        ),
        "active_row_deterministic_bounds": _active_row_taylor_bounds(),
        "row_peak_min": float(row_peaks.min()),
        "row_peak_max": float(row_peaks.max()),
        "all_rows": _summarize_taylor_rows(
            R,
            beta_hat,
            all_mask,
            total_linear=total_linear,
            total_quadratic=total_quadratic,
            total_remainder=total_remainder,
        ),
        "active_rows": _summarize_taylor_rows(
            R,
            beta_hat,
            active_mask,
            total_linear=total_linear,
            total_quadratic=total_quadratic,
            total_remainder=total_remainder,
        ),
        "inactive_rows": _summarize_taylor_rows(
            R,
            beta_hat,
            inactive_mask,
            total_linear=total_linear,
            total_quadratic=total_quadratic,
            total_remainder=total_remainder,
        ),
    }


def _beta_taylor_expectation_summary(
    mean_r: float,
    mean_r2: float,
    mean_remainder: float,
) -> Dict[str, Any]:
    if mean_r <= 0.0:
        return {
            "mean_r": None,
            "mean_r2": None,
            "mean_exact_remainder_unscaled": None,
            "quadratic_over_linear": None,
            "exact_remainder_over_linear": None,
            "second_order_remainder_over_linear": None,
            "quadratic_fraction_of_exact_remainder": None,
            "first_order_remainder_fraction_of_exact_prior": None,
            "second_order_remainder_fraction_of_exact_prior": None,
            "second_order_remainder_over_second_order_approx": None,
        }
    quadratic_over_linear = 0.5 * mean_r2 / mean_r
    exact_remainder_over_linear = mean_remainder / mean_r
    second_order_remainder_over_linear = (
        exact_remainder_over_linear - quadratic_over_linear
    )
    return {
        "mean_r": float(mean_r),
        "mean_r2": float(mean_r2),
        "mean_exact_remainder_unscaled": float(mean_remainder),
        "quadratic_over_linear": float(quadratic_over_linear),
        "exact_remainder_over_linear": float(exact_remainder_over_linear),
        "second_order_remainder_over_linear": float(
            second_order_remainder_over_linear
        ),
        "quadratic_fraction_of_exact_remainder": (
            float(quadratic_over_linear / exact_remainder_over_linear)
            if exact_remainder_over_linear > 0.0
            else None
        ),
        "first_order_remainder_fraction_of_exact_prior": float(
            exact_remainder_over_linear
            / (1.0 + exact_remainder_over_linear)
        ),
        "second_order_remainder_fraction_of_exact_prior": float(
            second_order_remainder_over_linear
            / (1.0 + exact_remainder_over_linear)
        ),
        "second_order_remainder_over_second_order_approx": float(
            second_order_remainder_over_linear
            / (1.0 + quadratic_over_linear)
        ),
    }


def compute_beta_activation_conditioned_taylor_expectations(
    beta: float,
    delta: float,
    samples_per_row: int,
    *,
    nominal_atoms: Optional[int] = None,
) -> Dict[str, Any]:
    """Analytic Taylor ratios conditional on a Beta row's active state."""

    beta = float(beta)
    delta = float(delta)
    samples_per_row = int(samples_per_row)
    if not np.isfinite(beta) or beta <= 1.0:
        raise ValueError("beta must be finite and greater than one.")
    if not np.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly inside (0, 1).")
    if samples_per_row < 1:
        raise ValueError("samples_per_row must be positive.")
    if nominal_atoms is not None and int(nominal_atoms) < 1:
        raise ValueError("nominal_atoms must be positive when supplied.")

    log_one_minus_delta = np.log1p(-delta)
    log_tail_probability = beta * log_one_minus_delta
    tail_probability = float(np.exp(log_tail_probability))
    cdf = float(-np.expm1(log_tail_probability))
    log_cdf = (
        float(np.log(cdf))
        if cdf < 0.5
        else float(np.log1p(-tail_probability))
    )
    log_probability_inactive = samples_per_row * log_cdf
    probability_inactive = float(np.exp(log_probability_inactive))
    probability_active = float(-np.expm1(log_probability_inactive))

    mean_r = 1.0 / (beta + 1.0)
    mean_r2 = 2.0 / ((beta + 1.0) * (beta + 2.0))
    mean_remainder = 1.0 / (beta * (beta + 1.0))

    (
        low_r_mass,
        low_r2_mass,
        low_remainder_mass,
    ) = _beta_low_partial_moments(
        beta,
        delta,
    )

    inactive = _beta_taylor_expectation_summary(
        low_r_mass / cdf,
        low_r2_mass / cdf,
        low_remainder_mass / cdf,
    )

    high_r_mass = (
        tail_probability
        * (1.0 + beta * delta)
        / (beta + 1.0)
    )
    high_r2_mass = (
        tail_probability
        * (
            2.0
            + 2.0 * beta * delta
            + beta * (beta + 1.0) * delta ** 2
        )
        / ((beta + 1.0) * (beta + 2.0))
    )
    high_log_mass = tail_probability * (
        1.0 / beta - log_one_minus_delta
    )
    high_remainder_mass = high_log_mass - high_r_mass
    probability_other_active = float(
        -np.expm1((samples_per_row - 1) * log_cdf)
    )
    if probability_active > 0.0:
        active = _beta_taylor_expectation_summary(
            (high_r_mass + probability_other_active * low_r_mass)
            / probability_active,
            (high_r2_mass + probability_other_active * low_r2_mass)
            / probability_active,
            (
                high_remainder_mass
                + probability_other_active * low_remainder_mass
            )
            / probability_active,
        )
    else:
        active = _beta_taylor_expectation_summary(0.0, 0.0, 0.0)

    for group, probability in (
        (inactive, probability_inactive),
        (active, probability_active),
    ):
        group["fraction_of_unconditional_linear_energy"] = (
            float(probability * group["mean_r"] / mean_r)
            if group["mean_r"] is not None
            else 0.0
        )
        group["fraction_of_unconditional_quadratic_energy"] = (
            float(probability * group["mean_r2"] / mean_r2)
            if group["mean_r2"] is not None
            else 0.0
        )
        group["fraction_of_unconditional_exact_remainder"] = (
            float(
                probability
                * group["mean_exact_remainder_unscaled"]
                / mean_remainder
            )
            if group["mean_exact_remainder_unscaled"] is not None
            else 0.0
        )

    return {
        "beta": beta,
        "delta": delta,
        "samples_per_row": samples_per_row,
        "nominal_atoms": (
            int(nominal_atoms) if nominal_atoms is not None else None
        ),
        "cdf_at_delta": float(cdf),
        "log_probability_row_inactive": float(log_probability_inactive),
        "log10_probability_row_inactive": float(
            log_probability_inactive / np.log(10.0)
        ),
        "probability_row_inactive": probability_inactive,
        "probability_row_active": probability_active,
        "inactive_row_deterministic_bounds": (
            _inactive_row_taylor_bounds(delta)
        ),
        "active_row_deterministic_bounds": _active_row_taylor_bounds(),
        "expected_active_rows": (
            float(int(nominal_atoms) * probability_active)
            if nominal_atoms is not None
            else None
        ),
        "expected_inactive_rows": (
            float(int(nominal_atoms) * probability_inactive)
            if nominal_atoms is not None
            else None
        ),
        "unconditional": _beta_taylor_expectation_summary(
            mean_r,
            mean_r2,
            mean_remainder,
        ),
        "inactive_rows": inactive,
        "active_rows": active,
    }


def compute_active_span_diagnostics(
    D: np.ndarray,
    X_norm: np.ndarray,
    R_activation: np.ndarray,
    R_effective: np.ndarray,
    delta: float,
    *,
    residual_tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """Compute the empirical restricted active-span ratios in Eq. 19."""

    D = _validate_matrix(D, "D")
    X_norm = _validate_matrix(X_norm, "X_norm")
    R_activation = _validate_matrix(R_activation, "R_activation")
    R_effective = _validate_matrix(R_effective, "R_effective")

    d, K = D.shape
    if X_norm.shape[0] != d:
        raise ValueError("D and X_norm disagree on feature dimension.")
    if R_activation.shape[0] != K or R_effective.shape[0] != K:
        raise ValueError("D and coefficient matrices disagree on atom dimension.")
    if X_norm.shape[1] != R_activation.shape[1] or X_norm.shape[1] != R_effective.shape[1]:
        raise ValueError("X_norm and coefficient matrices disagree on sample count.")
    if not np.isfinite(delta) or delta < 0.0 or delta >= 1.0:
        raise ValueError("delta must lie in [0, 1).")

    dictionary_column_norms = np.linalg.norm(D, axis=0)
    x_column_means = X_norm.mean(axis=0)
    x_column_stds = X_norm.std(axis=0)
    nonconstant_columns = x_column_stds > np.finfo(np.float64).eps
    integrity = {
        "dictionary_column_norm_max_abs_error": float(
            np.max(np.abs(dictionary_column_norms - 1.0))
        ),
        "x_column_mean_max_abs": float(np.max(np.abs(x_column_means))),
        "x_constant_column_count": int((~nonconstant_columns).sum()),
        "x_nonconstant_column_std_max_abs_error": (
            float(np.max(np.abs(x_column_stds[nonconstant_columns] - 1.0)))
            if np.any(nonconstant_columns)
            else None
        ),
    }

    active_mask = np.max(np.abs(R_activation), axis=1) > delta
    active_count = int(active_mask.sum())
    if active_count == 0:
        return {
            "condition_evaluable": False,
            "unevaluable_reason": "empty_active_set",
            "active_count": 0,
            "total_atoms": int(K),
            "active_fraction": 0.0,
            "valid_residual_count": None,
            "zero_residual_count": None,
            "alpha_empirical_min": None,
            "alpha_percentiles": {},
            "geometry_factor_active_count_over_alpha": None,
            "active_dictionary_rank": 0,
            "active_dictionary_columns": 0,
            "min_positive_singular_value": None,
            "max_singular_value": None,
            "positive_singular_condition_number": None,
            **integrity,
            "residual_tolerance": float(residual_tolerance),
        }

    D_active = D[:, active_mask]
    residual = X_norm - D.dot(R_effective)
    denominator = np.square(residual).sum(axis=0)
    valid = denominator > residual_tolerance
    if not np.any(valid):
        raise ValueError("All residuals are below residual_tolerance.")

    correlations = D_active.T.dot(residual[:, valid])
    numerator = np.square(correlations).sum(axis=0)
    alpha_values = numerator / denominator[valid]
    alpha_empirical = float(alpha_values.min())

    singular_values = np.linalg.svd(D_active, compute_uv=False)
    positive_singular_values = singular_values[
        singular_values > np.finfo(np.float64).eps * max(D_active.shape)
    ]
    min_positive_singular = (
        float(positive_singular_values.min())
        if positive_singular_values.size
        else 0.0
    )
    max_singular = float(singular_values.max()) if singular_values.size else 0.0
    return {
        "condition_evaluable": True,
        "unevaluable_reason": None,
        "active_count": active_count,
        "total_atoms": int(K),
        "active_fraction": float(active_count / float(K)),
        "valid_residual_count": int(valid.sum()),
        "zero_residual_count": int((~valid).sum()),
        "alpha_empirical_min": alpha_empirical,
        "alpha_percentiles": {
            str(p): float(value)
            for p, value in zip(
                (0, 1, 5, 25, 50, 75, 95, 99, 100),
                np.percentile(alpha_values, (0, 1, 5, 25, 50, 75, 95, 99, 100)),
            )
        },
        "geometry_factor_active_count_over_alpha": (
            float(active_count / alpha_empirical)
            if alpha_empirical > 0
            else None
        ),
        "active_dictionary_rank": int(np.linalg.matrix_rank(D_active)),
        "active_dictionary_columns": int(D_active.shape[1]),
        "min_positive_singular_value": min_positive_singular,
        "max_singular_value": max_singular,
        "positive_singular_condition_number": (
            float(max_singular / min_positive_singular)
            if min_positive_singular > 0
            else None
        ),
        **integrity,
        "residual_tolerance": float(residual_tolerance),
    }


def _flatten(prefix: str, value: Any) -> Iterable:
    if isinstance(value, dict):
        for key in sorted(value):
            next_prefix = "{}.{}".format(prefix, key) if prefix else str(key)
            for item in _flatten(next_prefix, value[key]):
                yield item
    elif isinstance(value, (list, tuple)):
        yield prefix, json.dumps(value)
    else:
        yield prefix, value


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return "{:.8g}".format(value)
    return str(value)


def render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# PADL theory-assumption diagnostics",
        "",
        "- Artifact stage: `{}`".format(report["artifact_stage"]),
        "- Artifact: `{}`".format(report["artifact_path"]),
        "",
        "## Taylor and small-coefficient diagnostics",
        "",
        "| Metric | Value |",
        "|:---|---:|",
    ]
    taylor = report["taylor"]
    ordered = (
        "beta_hat",
        "fitted_prior_mean",
        "mean",
        "std",
        "max_coefficient",
        "t1_linear",
        "t2_quadratic",
        "t2_over_t1",
        "t3_cubic",
        "t3_over_t1",
        "exact_energy",
        "exact_remainder",
        "exact_remainder_over_t1",
        "exact_remainder_over_exact",
        "exact_remainder_lower_bound",
        "exact_remainder_upper_bound",
        "exact_remainder_upper_bound_over_t1",
        "t2_over_exact_remainder",
        "second_order_remainder_over_exact",
    )
    for key in ordered:
        lines.append("| `{}` | {} |".format(key, _format_value(taylor.get(key))))

    lines.extend(["", "### Coefficient percentiles", "", "| Percentile | Value |", "|---:|---:|"])
    for key, value in taylor["percentiles"].items():
        lines.append("| {} | {} |".format(key, _format_value(value)))

    lines.extend(
        [
            "",
            "### Fractions at or below small-coefficient thresholds",
            "",
            "| Threshold | Fraction |",
            "|---:|---:|",
        ]
    )
    for key, value in taylor["fractions_at_or_below"].items():
        lines.append("| {} | {} |".format(key, _format_value(value)))

    lines.extend(
        [
            "",
            "### Pointwise first-order remainder",
            "",
            "For each coefficient this reports "
            "$-\\log(1-r)-r$ before multiplication by $\\widehat\\beta-1$.",
            "",
            "| Percentile | Value |",
            "|---:|---:|",
        ]
    )
    for key, value in taylor["pointwise_remainder_percentiles"].items():
        lines.append("| {} | {} |".format(key, _format_value(value)))

    regimes = report["taylor_regimes"]
    lines.extend(
        [
            "",
            "### Taylor energy by coefficient regime",
            "",
            "| Regime | Coefficient fraction | Linear-energy share | "
            "Remainder share | Quadratic / linear | Exact remainder / linear |",
            "|:---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in regimes["regimes"]:
        lines.append(
            "| {label} | {fraction} | {linear_share} | {remainder_share} | "
            "{quadratic_ratio} | {remainder_ratio} |".format(
                label=item["label"],
                fraction=_format_value(item["fraction_of_coefficients"]),
                linear_share=_format_value(item["fraction_of_total_linear"]),
                remainder_share=_format_value(
                    item["fraction_of_total_exact_remainder"]
                ),
                quadratic_ratio=_format_value(item["quadratic_over_linear"]),
                remainder_ratio=_format_value(
                    item["exact_remainder_over_linear"]
                ),
            )
        )

    active = report.get("active_span")
    if active is not None:
        lines.extend(
            [
                "",
                "## Restricted active-span diagnostics",
                "",
                "- Evaluation artifact: `{}`".format(
                    report.get("active_span_artifact_path")
                ),
                "",
                "| Metric | Value |",
                "|:---|---:|",
            ]
        )
        for key in (
            "condition_evaluable",
            "unevaluable_reason",
            "active_count",
            "total_atoms",
            "active_fraction",
            "valid_residual_count",
            "zero_residual_count",
            "alpha_empirical_min",
            "geometry_factor_active_count_over_alpha",
            "active_dictionary_rank",
            "active_dictionary_columns",
            "min_positive_singular_value",
            "max_singular_value",
            "positive_singular_condition_number",
            "dictionary_column_norm_max_abs_error",
            "x_column_mean_max_abs",
            "x_constant_column_count",
            "x_nonconstant_column_std_max_abs_error",
        ):
            lines.append("| `{}` | {} |".format(key, _format_value(active.get(key))))
        lines.extend(["", "### Empirical $\\alpha_i$ percentiles", "", "| Percentile | Value |", "|---:|---:|"])
        for key, value in active["alpha_percentiles"].items():
            lines.append("| {} | {} |".format(key, _format_value(value)))

    return "\n".join(lines) + "\n"


def load_and_compute(
    artifact_path: str,
    *,
    active_span_artifact_path: Optional[str] = None,
    residual_tolerance: float = 1e-12,
) -> Dict[str, Any]:
    path = Path(artifact_path)
    with np.load(path, allow_pickle=False) as artifact:
        required = {"stage", "R_warmup", "beta_hat"}
        missing = required.difference(artifact.files)
        if missing:
            raise ValueError("Artifact is missing required fields: {}".format(sorted(missing)))
        stage = _as_scalar_string(artifact["stage"])
        if stage != REQUIRED_STAGE:
            raise ValueError(
                "Formal diagnostics require stage={!r}; got {!r}.".format(
                    REQUIRED_STAGE, stage
                )
            )
        R_warmup = np.asarray(artifact["R_warmup"])
        beta_hat = _as_scalar_float(artifact["beta_hat"], "beta_hat")
        warmup_config = _as_config(
            artifact["config_json"], "warm-up config_json"
        ) if "config_json" in artifact.files else {}
        report: Dict[str, Any] = {
            "artifact_path": str(path.resolve()),
            "artifact_stage": stage,
            "artifact_config": warmup_config,
            "taylor": compute_taylor_diagnostics(R_warmup, beta_hat),
            "taylor_regimes": compute_taylor_regime_diagnostics(
                R_warmup,
                beta_hat,
            ),
        }

        legacy_active_fields = {"D", "X_norm", "R_effective", "delta"}
        present = legacy_active_fields.intersection(artifact.files)
        if present and present != legacy_active_fields:
            raise ValueError(
                "Active-span analysis requires all fields {}; present {}.".format(
                    sorted(legacy_active_fields), sorted(present)
                )
            )
        if active_span_artifact_path is not None and present:
            raise ValueError(
                "Supply either a separate active-span artifact or legacy "
                "combined fields, not both."
            )
        if present == legacy_active_fields:
            report["active_span"] = compute_active_span_diagnostics(
                artifact["D"],
                artifact["X_norm"],
                R_warmup,
                artifact["R_effective"],
                _as_scalar_float(artifact["delta"], "delta"),
                residual_tolerance=residual_tolerance,
            )
            report["active_span_artifact_path"] = str(path.resolve())
        else:
            report["active_span"] = None
            report["active_span_artifact_path"] = None

    if active_span_artifact_path is not None:
        active_path = Path(active_span_artifact_path)
        with np.load(active_path, allow_pickle=False) as active_artifact:
            required = {
                "stage",
                "D",
                "X_norm",
                "R_activation",
                "R_effective",
                "delta",
            }
            missing = required.difference(active_artifact.files)
            if missing:
                raise ValueError(
                    "Active-span artifact is missing required fields: {}".format(
                        sorted(missing)
                    )
                )
            stage = _as_scalar_string(active_artifact["stage"])
            if stage != ACTIVE_SPAN_STAGE:
                raise ValueError(
                    "Active-span diagnostics require stage={!r}; got {!r}.".format(
                        ACTIVE_SPAN_STAGE, stage
                    )
                )
            active_config = _as_config(
                active_artifact["config_json"], "active-span config_json"
            ) if "config_json" in active_artifact.files else {}
            _validate_matching_provenance(warmup_config, active_config)
            report["active_span"] = compute_active_span_diagnostics(
                active_artifact["D"],
                active_artifact["X_norm"],
                active_artifact["R_activation"],
                active_artifact["R_effective"],
                _as_scalar_float(active_artifact["delta"], "delta"),
                residual_tolerance=residual_tolerance,
            )
            report["active_span_artifact_path"] = str(active_path.resolve())
            report["active_span_artifact_config"] = active_config
    return report


def write_report(report: Dict[str, Any], output_dir: str) -> Path:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite output directory: {}".format(destination))
    destination.mkdir(parents=True)

    with (destination / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)

    with (destination / "diagnostics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in _flatten("", report):
            writer.writerow([key, value])

    (destination / "diagnostics.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a PADL warm-up artifact without running training."
    )
    parser.add_argument("--artifact", required=True, help="NPZ artifact path.")
    parser.add_argument(
        "--active-span-artifact",
        default=None,
        help=(
            "Optional matched evaluation NPZ containing D, X_norm, "
            "R_activation, R_effective, and delta."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New directory for JSON, CSV, and Markdown outputs.",
    )
    parser.add_argument("--residual-tolerance", type=float, default=1e-12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = load_and_compute(
        args.artifact,
        active_span_artifact_path=args.active_span_artifact,
        residual_tolerance=args.residual_tolerance,
    )
    destination = write_report(report, args.output_dir)
    print("Wrote diagnostics to {}".format(destination))


if __name__ == "__main__":
    main()
