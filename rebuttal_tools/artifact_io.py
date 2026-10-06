"""Helpers for saving PADL rebuttal diagnostic artifacts.

This module is intentionally independent of the repository's training classes.
Call ``save_warmup_artifact`` from the production Algorithm 1 implementation at
``t == t_s``, before gating and before re-initialization.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


WARMUP_STAGE = "warmup_pre_gate"
ACTIVE_SPAN_STAGE = "evaluation_coefficients"
# Backward-compatible alias used by the first staged version of the tool.
STAGE = WARMUP_STAGE


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def save_warmup_artifact(
    path: str,
    *,
    R_warmup: Any,
    beta_hat: float,
    D_warmup: Optional[Any] = None,
    X_warmup_norm: Optional[Any] = None,
    D: Optional[Any] = None,
    X_norm: Optional[Any] = None,
    R_effective: Optional[Any] = None,
    delta: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a non-overwriting NPZ artifact for rebuttal diagnostics.

    ``R_warmup`` must be the ungated coefficient matrix in ``[0, 1)`` at the
    one-shot warm-up snapshot. For active-span diagnostics, ``D``, ``X_norm``,
    ``R_effective`` and ``delta`` must all come from the same run and the same
    normalized coordinate system.
    """

    output = Path(path)
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing artifact: {}".format(output))
    output.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "stage": np.asarray(WARMUP_STAGE),
        "R_warmup": _to_numpy(R_warmup),
        "beta_hat": np.asarray(float(beta_hat), dtype=np.float64),
        "config_json": np.asarray(
            json.dumps(config or {}, ensure_ascii=False, sort_keys=True)
        ),
    }

    optional_values = {
        "D_warmup": D_warmup,
        "X_warmup_norm": X_warmup_norm,
        "D": D,
        "X_norm": X_norm,
        "R_effective": R_effective,
    }
    for key, value in optional_values.items():
        if value is not None:
            arrays[key] = _to_numpy(value)
    if delta is not None:
        arrays["delta"] = np.asarray(float(delta), dtype=np.float64)

    np.savez_compressed(output, **arrays)
    return output


def save_active_span_artifact(
    path: str,
    *,
    D: Any,
    X_norm: Any,
    R_activation: Any,
    R_effective: Any,
    delta: float,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save the matched evaluation state needed to check Eq. 19.

    ``R_activation`` defines the active set through
    ``max_i |R_ki| > delta``. ``R_effective`` is the coefficient matrix
    actually used in ``X_norm - D @ R_effective``.
    """

    output = Path(path)
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing artifact: {}".format(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        stage=np.asarray(ACTIVE_SPAN_STAGE),
        D=_to_numpy(D),
        X_norm=_to_numpy(X_norm),
        R_activation=_to_numpy(R_activation),
        R_effective=_to_numpy(R_effective),
        delta=np.asarray(float(delta), dtype=np.float64),
        config_json=np.asarray(
            json.dumps(config or {}, ensure_ascii=False, sort_keys=True)
        ),
    )
    return output
