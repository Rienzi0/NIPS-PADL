"""Append-only run-level metrics used by rebuttal experiments."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


RUN_KEY_FIELDS = ("dataset", "seed", "split_id", "method")
DEFAULT_FIELDS = (
    "dataset",
    "seed",
    "split_id",
    "method",
    "rmse",
    "psnr",
    "ssim",
    "patch_psnr",
    "proxy_psnr",
    "proxy_ssim",
    "avg_atoms",
    "activated_atoms",
    "calibration_seconds",
    "train_seconds",
    "predict_seconds",
    "total_seconds",
    "data_fingerprint",
    "config_json",
)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.ascontiguousarray(np.asarray(value))


def tensor_fingerprint(value: Any) -> str:
    """Hash shape, dtype, and exact bytes of a dataset tensor."""

    array = _to_numpy(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def normalize_run_row(row: Dict[str, Any]) -> Dict[str, str]:
    missing = set(RUN_KEY_FIELDS).difference(row)
    if missing:
        raise ValueError("Run row is missing key fields: {}".format(sorted(missing)))
    normalized: Dict[str, str] = {}
    for field in DEFAULT_FIELDS:
        value = row.get(field, "")
        if field == "config_json" and isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        normalized[field] = "" if value is None else str(value)
    return normalized


def read_run_rows(path: str) -> List[Dict[str, str]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if reader.fieldnames != list(DEFAULT_FIELDS):
        raise ValueError(
            "Existing CSV schema does not match expected fields: {}".format(
                list(DEFAULT_FIELDS)
            )
        )
    return rows


def run_key(row: Dict[str, Any]) -> Sequence[str]:
    return tuple(str(row[field]) for field in RUN_KEY_FIELDS)


def run_exists(path: str, row: Dict[str, Any]) -> bool:
    target = run_key(row)
    return any(run_key(existing) == target for existing in read_run_rows(path))


def append_run_row(path: str, row: Dict[str, Any]) -> Path:
    """Append one run atomically at row granularity; refuse duplicate keys."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_run_row(row)
    existing = read_run_rows(str(destination))
    target = run_key(normalized)
    if any(run_key(item) == target for item in existing):
        raise ValueError("Duplicate run key {} in {}".format(target, destination))

    write_header = not destination.exists()
    with destination.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DEFAULT_FIELDS))
        if write_header:
            writer.writeheader()
        writer.writerow(normalized)
        handle.flush()
    return destination
