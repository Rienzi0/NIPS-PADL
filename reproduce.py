#!/usr/bin/env python3
"""Run the confirmed historical CIFAR experiment or its fixed-configuration SVHN extension."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from padl.experiment import run_experiment


ROOT = Path(__file__).resolve().parent


def tensor_fingerprint(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar100", "svhn"), default="cifar100")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, required=True,
                        help="New result directory; existing directories are rejected")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto",
                        help="auto selects CUDA, then MPS, then CPU; the recorded historical run used MPS")
    parser.add_argument("--threads", type=int, default=1)
    return parser


def resolve_device(device):
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    return device


def run(output, data_dir, device="auto", download=False, dataset="cifar100"):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory: {}".format(output))
    if dataset not in ("cifar100", "svhn"):
        raise ValueError("Dataset must be cifar100 or svhn")
    device = resolve_device(device)
    profile = json.loads((ROOT / "configs" / (dataset + ".json")).read_text())
    print("{}: seed {}, {}/{} random patches, beta {}, delta {}, device {}".format(
        dataset, profile["seed"], profile["train_patches"], profile["test_patches"],
        profile["beta_initialization"], profile["threshold"], device), flush=True)
    started = time.perf_counter()
    metrics, arrays = run_experiment(profile, device, str(data_dir), download)
    elapsed = time.perf_counter() - started
    fingerprint = tensor_fingerprint(np.concatenate(
        [arrays["X_train_raw"], arrays["X_test_raw"]], axis=1))
    row = {
        "dataset": dataset, "method": "historical_direct_gate",
        "split_id": profile["split_id"], "data_fingerprint": fingerprint,
        "dictionary_fingerprint": tensor_fingerprint(arrays["D"]),
        **metrics, "device": device, "elapsed_seconds": elapsed,
    }
    reference = profile.get("reference_metrics")
    errors = {key: abs(metrics[key] - value) for key, value in reference.items()} if reference else None
    matches = all(errors[key] <= profile["comparison_tolerances"][key] for key in errors) if reference else None
    expected_fingerprint = profile.get("data_fingerprint")
    data_matches = fingerprint == expected_fingerprint if expected_fingerprint else None
    if data_matches is False:
        matches = False
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "numpy", "scipy", "Pillow", "torchmetrics")}
    source_paths = sorted(list(ROOT.glob("*.py")) + list((ROOT / "padl").glob("*.py")))
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in source_paths}
    manifest = {
        "profile": profile, "run": row,
        "reference_metric_errors": errors, "matches_reference": matches,
        "matches_data_fingerprint": data_matches,
        "independent_training_runs": 1, "paper_table2_repeated_runs_reproduced": False,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "device": device, "threads": torch.get_num_threads(), "packages": versions},
        "source_sha256": sources,
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "metrics.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with (output / "matched_evaluation.npz").open("xb") as handle:
        np.savez_compressed(handle, **arrays,
                            stage=np.asarray("historical_direct_gated_evaluation"),
                            config_json=np.asarray(json.dumps(manifest, sort_keys=True)))
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(row, sort_keys=True), flush=True)
    return manifest


def main():
    parser = build_argparser()
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        manifest = run(args.output, args.data_dir, args.device, args.download, args.dataset)
    except (ValueError, FileExistsError, RuntimeError, OSError) as error:
        parser.exit(2, "Reproduction failed: {}\n".format(error))
    if manifest["matches_reference"] is False:
        parser.exit(1, "Actual results saved; data or metrics differ from the recorded reference. "
                    "See manifest.json for the differences and runtime.\n")
    if manifest["matches_reference"] is None:
        print("Measurements saved; this configuration has no established numerical reference yet.", flush=True)
    else:
        print("Results match the recorded reference within the declared tolerances.", flush=True)


if __name__ == "__main__":
    main()
