#!/usr/bin/env python3
"""Auditable Algorithm 1 PADL reference runner for rebuttal experiments.

This runner is intentionally separate from the repository's legacy
``L_inf_dic_*`` implementations. It follows Algorithm 1 in the submitted
paper:

1. optimize a warm-up objective with ``(lambda1, lambda2) = (0, 1)``;
2. fit ``Beta(1, beta)`` to the ungated warm-up coefficients;
3. set ``lambda1 = beta - 1``, ``lambda2 = 1`` and solve Eq. 13;
4. re-initialize ``D`` and ``R``;
5. optimize the practical row-wise PADL objective;
6. keep ``D`` fixed at test time and optimize only ``R``.

The training objective never hard-gates coefficients. ``delta`` defines the
reported active set ``max_i |R_ki| > delta``, matching Algorithm 1 and Eq. 8.
It must pass a real-data reproduction smoke test before replacing the
historical experiment path.
"""

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

try:
    from .paper_patch_data import load_paper_patch_splits
except ImportError:  # Direct script execution after deployment to repo root.
    from paper_patch_data import load_paper_patch_splits
from rebuttal_tools.artifact_io import (
    save_active_span_artifact,
    save_warmup_artifact,
)
from rebuttal_tools.run_metrics_io import (
    append_run_row,
    run_exists,
    tensor_fingerprint,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_columns(
    X: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize each sample column and return ``(X_norm, mean, std)``."""

    mean = X.mean(dim=0, keepdim=True)
    raw_std = X.std(dim=0, keepdim=True, unbiased=False)
    std = raw_std.clamp_min(eps)
    normalized = (X - mean) / std
    normalized = torch.where(
        raw_std < eps,
        torch.zeros_like(normalized),
        normalized,
    )
    return normalized, mean, std


def initialize_dictionary(
    d: int,
    K: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    D = torch.rand(d, K, device=device, dtype=dtype)
    D = D / D.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return D.requires_grad_(True)


def initialize_coefficients(
    K: int,
    N: int,
    beta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if beta <= 0:
        raise ValueError("beta must be positive.")
    concentration1 = torch.tensor(1.0, device=device, dtype=dtype)
    concentration0 = torch.tensor(float(beta), device=device, dtype=dtype)
    distribution = torch.distributions.Beta(concentration1, concentration0)
    R = distribution.sample((K, N))
    return R.clamp_(0.0, 1.0 - 1e-7).requires_grad_(True)


def rowwise_linf(R: torch.Tensor) -> torch.Tensor:
    """Return ``sum_k max_i |R_ki|`` for ``R`` shaped ``[K, N]``."""

    if R.ndim != 2:
        raise ValueError("R must be two-dimensional [K, N].")
    return torch.max(torch.abs(R), dim=1).values.sum()


def objective_components(
    X: torch.Tensor,
    D: torch.Tensor,
    R: torch.Tensor,
    lambda1: float,
    lambda2: float,
) -> Dict[str, torch.Tensor]:
    residual = X - D.matmul(R)
    reconstruction = residual.square().sum()
    l1 = torch.abs(R).sum()
    linf = rowwise_linf(R)
    total = reconstruction + float(lambda1) * l1 + float(lambda2) * linf
    return {
        "total": total,
        "reconstruction": reconstruction,
        "l1": l1,
        "linf": linf,
    }


def fit_beta1_mle(R: torch.Tensor, eps: float = 1e-7) -> float:
    """Closed-form MLE for ``Beta(1, beta)`` with support fixed to [0, 1]."""

    values = R.detach().double().clamp(min=0.0, max=1.0 - eps)
    denominator = float((-torch.log1p(-values)).sum().item())
    if denominator <= 0:
        raise ValueError("Cannot fit beta: all coefficients are numerically zero.")
    beta_hat = float(values.numel() / denominator)
    if not np.isfinite(beta_hat) or beta_hat <= 1.0:
        raise ValueError(
            "Fitted beta must be > 1 for the stated PADL calibration; got "
            "{:.8g}.".format(beta_hat)
        )
    return beta_hat


def concentration_u(N: int, beta: float, tau: float) -> float:
    """The ``U(N, beta, tau)`` quantity stated in Theorem 2.4."""

    if N <= 0:
        raise ValueError("N must be positive.")
    if beta <= 0:
        raise ValueError("beta must be positive.")
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie in (0, 1).")
    return float(
        N / (1.0 + beta)
        + math.sqrt(N / 2.0 * math.log(1.0 / tau))
    )


def _log_one_minus_exp(log_x: float) -> float:
    """Stable ``log(1-exp(log_x))`` for ``log_x <= 0``."""

    if log_x >= 0.0:
        return float("-inf")
    if log_x < math.log(0.5):
        return math.log1p(-math.exp(log_x))
    return math.log(-math.expm1(log_x))


def delta_stationarity_residual(
    delta: float,
    *,
    N: int,
    beta: float,
    lambda1: float,
    lambda2: float,
    tau: float,
) -> float:
    """Left minus right side of Eq. 13, evaluated stably."""

    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly inside (0, 1).")
    U = concentration_u(N, beta, tau)
    one_minus_delta = 1.0 - delta
    F_delta = 1.0 - one_minus_delta ** beta
    if not 0.0 < F_delta < 1.0:
        raise ValueError("F(delta) is numerically outside (0, 1).")
    log_F = math.log(F_delta)
    log_F_power_N = N * log_F
    log_denominator = _log_one_minus_exp(log_F_power_N)
    log_f = math.log(beta) + (beta - 1.0) * math.log(one_minus_delta)
    log_left = (
        math.log(float(N))
        + (N - 1.0) * log_F
        + log_f
        - log_denominator
    )
    left = math.exp(log_left) if log_left > -745.0 else 0.0
    right = (
        2.0
        * float(lambda2)
        / (
            U
            * (
                float(lambda1)
                + float(lambda2) * float(delta) / U
            )
        )
    )
    return float(left - right)


def solve_optimal_delta(
    *,
    N: int,
    beta: float,
    lambda1: float,
    lambda2: float = 1.0,
    tau: float = 0.05,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> float:
    """Solve the interior stationary condition in Eq. 13 by bracketing."""

    if lambda1 <= 0 or lambda2 <= 0:
        raise ValueError("lambda1 and lambda2 must be positive.")

    # The root can be close to one for large N. Combine a linear grid with a
    # log-distance-to-one grid before bisection.
    linear = np.linspace(1e-6, 1.0 - 1e-6, 4096)
    near_one = 1.0 - np.logspace(-12, -3, 2048)
    candidates = np.unique(np.concatenate([linear, near_one]))
    previous_x = float(candidates[0])
    previous_value = delta_stationarity_residual(
        previous_x,
        N=N,
        beta=beta,
        lambda1=lambda1,
        lambda2=lambda2,
        tau=tau,
    )
    bracket = None
    for value in candidates[1:]:
        x = float(value)
        current = delta_stationarity_residual(
            x,
            N=N,
            beta=beta,
            lambda1=lambda1,
            lambda2=lambda2,
            tau=tau,
        )
        if current == 0.0 or previous_value * current < 0.0:
            bracket = (previous_x, x)
            break
        previous_x, previous_value = x, current
    if bracket is None:
        raise RuntimeError(
            "Eq. 13 has no bracketed interior root for "
            "N={}, beta={:.8g}, lambda1={:.8g}, lambda2={:.8g}, tau={:.8g}."
            .format(N, beta, lambda1, lambda2, tau)
        )

    lo, hi = bracket
    g_lo = delta_stationarity_residual(
        lo,
        N=N,
        beta=beta,
        lambda1=lambda1,
        lambda2=lambda2,
        tau=tau,
    )
    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        g_mid = delta_stationarity_residual(
            mid,
            N=N,
            beta=beta,
            lambda1=lambda1,
            lambda2=lambda2,
            tau=tau,
        )
        if abs(g_mid) <= tolerance or (hi - lo) <= tolerance:
            return float(mid)
        if g_lo * g_mid <= 0.0:
            hi = mid
        else:
            lo, g_lo = mid, g_mid
    return float(0.5 * (lo + hi))


@dataclass
class PADLConfig:
    dict_size: int = 128
    outer_iters: int = 400
    warmup_steps: int = 50
    r_steps: int = 20
    d_steps: int = 10
    predict_steps: int = 100
    lr_r: float = 5e-3
    lr_d: float = 5e-4
    confidence_tau: float = 0.05
    effective_n: Optional[int] = None
    log_every: int = 20


class PaperPADL(object):
    """Projected alternating optimizer for the paper's practical objective."""

    def __init__(
        self,
        d_size: int,
        config: PADLConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if config.warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive.")
        if config.outer_iters <= config.warmup_steps:
            raise ValueError("outer_iters must be greater than warmup_steps.")
        self.d_size = int(d_size)
        self.config = config
        self.device = device
        self.dtype = dtype
        self.D: Optional[torch.Tensor] = None
        self.beta_hat: Optional[float] = None
        self.lambda1: Optional[float] = None
        self.lambda2: float = 1.0
        self.delta: Optional[float] = None
        self.calibration_seconds: Optional[float] = None
        self.train_seconds: Optional[float] = None

    def _new_state(
        self,
        N: int,
        beta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        D = initialize_dictionary(
            self.d_size,
            self.config.dict_size,
            self.device,
            self.dtype,
        )
        R = initialize_coefficients(
            self.config.dict_size,
            N,
            beta,
            self.device,
            self.dtype,
        )
        return D, R

    def _run_outer_steps(
        self,
        X_norm: torch.Tensor,
        D: torch.Tensor,
        R: torch.Tensor,
        *,
        steps: int,
        lambda1: float,
        lambda2: float,
        phase: str,
    ) -> None:
        optimizer_r = torch.optim.Adam([R], lr=self.config.lr_r)
        optimizer_d = torch.optim.Adam([D], lr=self.config.lr_d)

        for outer in range(1, steps + 1):
            for _ in range(self.config.r_steps):
                optimizer_r.zero_grad()
                components = objective_components(
                    X_norm,
                    D.detach(),
                    R,
                    lambda1,
                    lambda2,
                )
                components["total"].backward()
                optimizer_r.step()
                with torch.no_grad():
                    R.clamp_(0.0, 1.0 - 1e-7)

            for _ in range(self.config.d_steps):
                optimizer_d.zero_grad()
                residual = X_norm - D.matmul(R.detach())
                residual.square().sum().backward()
                optimizer_d.step()
                with torch.no_grad():
                    D.div_(D.norm(dim=0, keepdim=True).clamp_min(1e-12))

            if (
                self.config.log_every > 0
                and (outer == 1 or outer == steps or outer % self.config.log_every == 0)
            ):
                with torch.no_grad():
                    logged = objective_components(
                        X_norm, D, R, lambda1, lambda2
                    )
                print(
                    "[{} {:03d}/{:03d}] recon={:.6g} l1={:.6g} "
                    "row_linf={:.6g} total={:.6g}".format(
                        phase,
                        outer,
                        steps,
                        float(logged["reconstruction"].item()),
                        float(logged["l1"].item()),
                        float(logged["linf"].item()),
                        float(logged["total"].item()),
                    )
                )

    def fit(
        self,
        X: torch.Tensor,
        *,
        warmup_artifact_path: Optional[str] = None,
        artifact_config: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        started = time.perf_counter()
        X_norm, _, _ = normalize_columns(X)
        N = int(X_norm.shape[1])

        D, R = self._new_state(N, beta=1.0)
        self._run_outer_steps(
            X_norm,
            D,
            R,
            steps=self.config.warmup_steps,
            lambda1=0.0,
            lambda2=1.0,
            phase="warmup",
        )

        calibration_started = time.perf_counter()
        beta_hat = fit_beta1_mle(R)
        lambda1 = beta_hat - 1.0
        effective_n = self.config.effective_n or N
        delta = solve_optimal_delta(
            N=int(effective_n),
            beta=beta_hat,
            lambda1=lambda1,
            lambda2=1.0,
            tau=self.config.confidence_tau,
        )
        self.calibration_seconds = time.perf_counter() - calibration_started

        if warmup_artifact_path:
            metadata = dict(artifact_config or {})
            metadata.update(
                {
                    "config": asdict(self.config),
                    "completed_warmup_updates": int(
                        self.config.warmup_steps
                    ),
                    "coefficient_shape": [
                        int(R.shape[0]),
                        int(R.shape[1]),
                    ],
                    "objective_scaling": "unnormalized Eq. 8 terms",
                    "effective_n": int(effective_n),
                    "lambda1": float(lambda1),
                    "lambda2": 1.0,
                    "delta": float(delta),
                }
            )
            save_warmup_artifact(
                warmup_artifact_path,
                R_warmup=R,
                beta_hat=beta_hat,
                D_warmup=D,
                X_warmup_norm=X_norm,
                config=metadata,
            )

        print(
            "[calibration] beta={:.8g} lambda1={:.8g} lambda2=1 "
            "delta={:.8g} effective_n={}".format(
                beta_hat, lambda1, delta, effective_n
            )
        )

        # Algorithm 1 explicitly re-initializes both variables after the
        # one-shot estimate.
        D, R = self._new_state(N, beta=beta_hat)
        final_steps = self.config.outer_iters - self.config.warmup_steps
        self._run_outer_steps(
            X_norm,
            D,
            R,
            steps=final_steps,
            lambda1=lambda1,
            lambda2=1.0,
            phase="final",
        )

        self.D = D.detach()
        self.beta_hat = float(beta_hat)
        self.lambda1 = float(lambda1)
        self.delta = float(delta)
        self.train_seconds = time.perf_counter() - started
        return self.D

    def predict(self, X: torch.Tensor) -> Dict[str, Any]:
        if self.D is None or self.beta_hat is None or self.lambda1 is None:
            raise RuntimeError("Call fit before predict.")
        X_norm, mean, std = normalize_columns(X)
        N = int(X_norm.shape[1])
        R = initialize_coefficients(
            self.config.dict_size,
            N,
            self.beta_hat,
            self.device,
            self.dtype,
        )
        optimizer_r = torch.optim.Adam([R], lr=self.config.lr_r)
        started = time.perf_counter()
        for _ in range(self.config.predict_steps):
            optimizer_r.zero_grad()
            components = objective_components(
                X_norm,
                self.D,
                R,
                self.lambda1,
                self.lambda2,
            )
            components["total"].backward()
            optimizer_r.step()
            with torch.no_grad():
                R.clamp_(0.0, 1.0 - 1e-7)

        with torch.no_grad():
            R_final = R.detach()
            reconstruction_norm = self.D.matmul(R_final)
            reconstruction = reconstruction_norm * std + mean
            active_mask = torch.max(torch.abs(R_final), dim=1).values > float(
                self.delta
            )
        return {
            "D": self.D,
            "R": R_final,
            "X_norm": X_norm.detach(),
            "X_reconstruction": reconstruction.detach(),
            "active_mask": active_mask.detach(),
            "predict_seconds": time.perf_counter() - started,
        }


def rmse(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((reconstruction - target).square())).item())


def patch_psnr(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((reconstruction - target).square())
    if float(mse.item()) == 0.0:
        return float("inf")
    data_range = float((target.max() - target.min()).item())
    if data_range <= 0.0:
        return float("nan")
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(float(mse.item())))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the paper-aligned PADL reference implementation."
    )
    parser.add_argument("--dataset", choices=["cifar100", "svhn"], default="cifar100")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--train-patches", type=int, default=20000)
    parser.add_argument("--test-patches", type=int, default=2000)
    parser.add_argument("--dict-size", type=int, default=128)
    parser.add_argument("--outer-iters", type=int, default=400)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--r-steps", type=int, default=20)
    parser.add_argument("--d-steps", type=int, default=10)
    parser.add_argument("--predict-steps", type=int, default=100)
    parser.add_argument("--lr-r", type=float, default=5e-3)
    parser.add_argument("--lr-d", type=float, default=5e-4)
    parser.add_argument("--confidence-tau", type=float, default=0.05)
    parser.add_argument(
        "--effective-n",
        type=int,
        default=None,
        help=(
            "N used in Eq. 13. Defaults to the number of training patches. "
            "Any override is recorded and must be justified."
        ),
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--split-id", default="paper_nonoverlap_v1")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    method = "padl"
    run_key = {
        "dataset": args.dataset,
        "seed": args.seed,
        "split_id": args.split_id,
        "method": method,
    }
    if args.output_csv and run_exists(args.output_csv, run_key):
        if args.resume:
            print("Run already recorded; skipping {}".format(run_key))
            return
        raise ValueError(
            "Run already exists. Use --resume to skip: {}".format(run_key)
        )

    train_X, test_X = load_paper_patch_splits(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        train_patches=args.train_patches,
        test_patches=args.test_patches,
        patch_size=args.patch_size,
        seed=args.seed,
        device=device,
        download=args.download,
    )
    data_fingerprint = tensor_fingerprint(
        torch.cat([train_X.detach().cpu(), test_X.detach().cpu()], dim=1)
    )
    config = PADLConfig(
        dict_size=args.dict_size,
        outer_iters=args.outer_iters,
        warmup_steps=args.warmup_steps,
        r_steps=args.r_steps,
        d_steps=args.d_steps,
        predict_steps=args.predict_steps,
        lr_r=args.lr_r,
        lr_d=args.lr_d,
        confidence_tau=args.confidence_tau,
        effective_n=args.effective_n,
        log_every=args.log_every,
    )
    model = PaperPADL(
        d_size=args.patch_size * args.patch_size,
        config=config,
        device=device,
    )

    warmup_path = None
    active_path = None
    if args.artifact_dir:
        artifact_dir = Path(args.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        prefix = "{}_seed{}".format(args.dataset, args.seed)
        warmup_path = str(artifact_dir / (prefix + "_warmup.npz"))
        active_path = str(artifact_dir / (prefix + "_evaluation.npz"))

    provenance = {
        "dataset": args.dataset,
        "seed": args.seed,
        "split_id": args.split_id,
        "data_fingerprint": data_fingerprint,
        "train_patches": args.train_patches,
        "test_patches": args.test_patches,
        "patch_size": args.patch_size,
    }
    model.fit(
        train_X,
        warmup_artifact_path=warmup_path,
        artifact_config=provenance,
    )
    prediction = model.predict(test_X)
    result_rmse = rmse(prediction["X_reconstruction"], test_X)
    result_psnr = patch_psnr(prediction["X_reconstruction"], test_X)
    activated_atoms = int(prediction["active_mask"].sum().item())
    avg_atoms = float(
        (prediction["R"] > float(model.delta))
        .sum(dim=0)
        .float()
        .mean()
        .item()
    )

    if active_path:
        active_config = dict(provenance)
        active_config.update(
            {
                "lambda1": model.lambda1,
                "lambda2": model.lambda2,
                "delta": model.delta,
                "beta_hat": model.beta_hat,
            }
        )
        save_active_span_artifact(
            active_path,
            D=prediction["D"],
            X_norm=prediction["X_norm"],
            R_activation=prediction["R"],
            R_effective=prediction["R"],
            delta=float(model.delta),
            config=active_config,
        )

    total_seconds = float(model.train_seconds or 0.0) + float(
        prediction["predict_seconds"]
    )
    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "method": method,
        "rmse": result_rmse,
        "patch_psnr": result_psnr,
        "activated_atoms": activated_atoms,
        "avg_atoms": avg_atoms,
        "beta_hat": model.beta_hat,
        "lambda1": model.lambda1,
        "lambda2": model.lambda2,
        "delta": model.delta,
        "calibration_seconds": model.calibration_seconds,
        "train_seconds": model.train_seconds,
        "predict_seconds": prediction["predict_seconds"],
        "total_seconds": total_seconds,
        "data_fingerprint": data_fingerprint,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_csv:
        append_run_row(
            args.output_csv,
            {
                **run_key,
                "rmse": result_rmse,
                "patch_psnr": result_psnr,
                "avg_atoms": avg_atoms,
                "activated_atoms": activated_atoms,
                "calibration_seconds": model.calibration_seconds,
                "train_seconds": model.train_seconds,
                "predict_seconds": prediction["predict_seconds"],
                "total_seconds": total_seconds,
                "data_fingerprint": data_fingerprint,
                "config_json": {
                    **asdict(config),
                    "beta_hat": model.beta_hat,
                    "lambda1": model.lambda1,
                    "lambda2": model.lambda2,
                    "delta": model.delta,
                    "train_patches": args.train_patches,
                    "test_patches": args.test_patches,
                    "patch_size": args.patch_size,
                },
            },
        )
        print("Appended run metrics to {}".format(args.output_csv))


if __name__ == "__main__":
    main()
