#!/usr/bin/env python3
"""Reproduce versioned direct-gated image experiments with explicit provenance.

The default evaluates a released dictionary. The direct-gated implementation
is a historical variant, separate from the ungated Algorithm 1 reference.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from legacy_direct_gate import R_loss_delta, _clamp_R, _init_R_beta, dic_learning
from paper_patch_data import load_paper_patch_splits
from rebuttal_tools.run_metrics_io import tensor_fingerprint


def load_profile(name):
    if name not in ("cifar100", "svhn"):
        raise ValueError("Unknown profile: {}".format(name))
    with (ROOT / "profiles" / (name + ".json")).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_dictionary(profile, root=ROOT):
    path = Path(root) / profile["checkpoint"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != profile["checkpoint_sha256"]:
        raise ValueError("Dictionary checkpoint checksum mismatch: {}".format(path))
    with np.load(str(path), allow_pickle=False) as saved:
        array = saved["D"].copy()
    expected_shape = (profile["patch_size"] ** 2, profile["dict_size"])
    if array.shape != expected_shape or array.dtype != np.float32:
        raise ValueError("Dictionary must be float32 with shape {}".format(expected_shape))
    if not np.isfinite(array).all():
        raise ValueError("Dictionary entries must be finite")
    return torch.from_numpy(array)


def normalize(samples):
    mean = samples.mean(dim=0, keepdim=True)
    # Preserve the historical variant's sample std (correction=1).
    std = samples.std(dim=0, keepdim=True, unbiased=True) + 1e-8
    return (samples - mean) / std, mean, std


def encode(dictionary, normalized, profile, seed):
    torch.manual_seed(int(seed))
    logits = _init_R_beta(dictionary.shape[1], normalized.shape[1],
                          beta=profile["beta_initialization"], device=dictionary.device)
    logits.requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=profile["lr_r"])
    started = time.perf_counter()
    for _ in range(profile["predict_steps"]):
        optimizer.zero_grad()
        loss = R_loss_delta(normalized, logits, dictionary,
                            delta=profile["threshold"], tau=profile["tau"],
                            beta_weight=profile["beta_weight"],
                            include_beta_reg=profile["include_beta_reg"])
        loss.backward()
        optimizer.step()
    return _clamp_R(logits).detach(), time.perf_counter() - started


def evaluate(dictionary, samples, coefficients, threshold):
    normalized, mean, std = normalize(samples)
    effective = coefficients * (coefficients >= threshold).to(coefficients.dtype)
    reconstruction = (dictionary @ effective) * std + mean
    counts = (effective != 0).sum(dim=0)
    return {
        "rmse": float((samples - reconstruction).square().mean().sqrt()),
        "global_active_atoms": int((effective.abs().max(dim=1).values > 0).sum()),
        "mean_retained_per_patch": float(counts.float().mean()),
        "X_test_norm": normalized, "x_mean": mean, "x_std": std,
        "R_effective": effective, "X_reconstruction": reconstruction,
    }


def run(profile, train, test, output, mode="checkpoint", coefficient_seeds=None, device="cpu"):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory; refusing to overwrite {}".format(output))
    seeds = tuple((profile["default_coefficient_seed"],) if coefficient_seeds is None else coefficient_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Coefficient seeds must be non-empty and unique")
    if any(seed not in profile["coefficient_seeds"] for seed in seeds):
        raise ValueError("Use the coefficient seeds recorded in this versioned profile")
    if mode not in ("checkpoint", "train"):
        raise ValueError("Mode must be checkpoint or train")
    for samples, count in ((train, profile["train_patches"]), (test, profile["test_patches"])):
        if tuple(samples.shape) != (profile["patch_size"] ** 2, count):
            raise ValueError("Data shape does not match the profile")
        if samples.dtype != torch.float32 or not bool(torch.isfinite(samples).all()):
            raise ValueError("Data must be finite float32 patch matrices")
    fingerprint = tensor_fingerprint(torch.cat([train.cpu(), test.cpu()], dim=1))
    if fingerprint != profile["data_fingerprint"]:
        raise ValueError("Data fingerprint differs from the published profile; check dataset, split and preprocessing")
    dictionary = load_dictionary(profile) if mode == "checkpoint" else None
    output.mkdir(parents=True, exist_ok=False)
    runtime_device = torch.device(device)
    train, test = train.to(runtime_device), test.to(runtime_device)
    train_seconds = 0.0
    if mode == "train":
        torch.manual_seed(profile["dictionary_seed"])
        model = dic_learning(train.shape[0], profile["dict_size"], profile["fit_iters"],
                             runtime_device, delta=profile["threshold"], tau=profile["tau"],
                             beta=profile["training_beta_initialization"],
                             beta_weight=profile["beta_weight"], include_beta_reg=profile["include_beta_reg"])
        model.k_r, model.k_d = profile["r_steps"], profile["d_steps"]
        started = time.perf_counter()
        dictionary = model.fit(train).detach()
        train_seconds = time.perf_counter() - started
    else:
        dictionary = dictionary.to(runtime_device)
    dictionary_fingerprint = tensor_fingerprint(dictionary)
    normalized, _, _ = normalize(test)
    rows = []
    for seed in seeds:
        coefficients, seconds = encode(dictionary, normalized, profile, seed)
        state = evaluate(dictionary, test, coefficients, profile["threshold"])
        expected = profile["reference_metrics"][str(seed)]
        errors = {key: abs(state[key] - expected[key]) for key in expected}
        match = all(errors[key] <= profile["comparison_tolerances"][key] for key in expected)
        row = {
            "dataset": profile["dataset"], "method": profile["model_variant"],
            "mode": mode, "dictionary_seed": profile["dictionary_seed"],
            "coefficient_seed": seed, "split_id": profile["split_id"],
            "data_fingerprint": fingerprint, "dictionary_fingerprint": dictionary_fingerprint,
            "rmse": state["rmse"], "mean_retained_per_patch": state["mean_retained_per_patch"],
            "global_active_atoms": state["global_active_atoms"],
            "train_seconds": train_seconds, "predict_seconds": seconds,
            "matches_checkpoint_reference": match,
        }
        rows.append(row)
        arrays = {key: value.detach().cpu().numpy() for key, value in state.items() if torch.is_tensor(value)}
        with (output / ("evaluation_seed{}.npz".format(seed))).open("xb") as handle:
            np.savez_compressed(handle, D=dictionary.cpu().numpy(), X_test_raw=test.cpu().numpy(),
                                R_raw=coefficients.cpu().numpy(),
                                stage=np.asarray("direct_gated_variant_evaluation"),
                                config_json=np.asarray(json.dumps({"profile": profile, "run": row}, sort_keys=True)),
                                **arrays)
        print(json.dumps(row, sort_keys=True), flush=True)
    with (output / "metrics.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(ROOT.rglob("*.py"))
                     if ".venv" not in path.parts and "outputs" not in path.parts}
    versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "scipy", "Pillow")}
    manifest = {
        "profile": profile, "mode": mode, "coefficient_seeds": seeds,
        "dictionary_shared_across_repeats": True, "evaluation_samples_shared_across_repeats": True,
        "independent_dictionary_training_runs": 1 if mode == "train" else 0,
        "matches_checkpoint_reference": all(row["matches_checkpoint_reference"] for row in rows),
        "reference_errors": [{key: abs(row[key] - profile["reference_metrics"][str(row["coefficient_seed"])][key])
                              for key in profile["reference_metrics"][str(row["coefficient_seed"])]} for row in rows],
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "machine": platform.machine(), "device": str(runtime_device), "threads": torch.get_num_threads(), "packages": versions},
        "source_sha256": source_hashes, "runs": rows,
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("cifar100", "svhn"), default="cifar100")
    parser.add_argument("--mode", choices=("checkpoint", "train"), default="checkpoint")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="A new directory; existing paths are rejected")
    parser.add_argument("--all-seeds", action="store_true", help="All five coefficient seeds, sharing one dictionary and test set")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.exists():
        parser.error("--output already exists; select a new directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    torch.set_num_threads(args.threads)
    profile = load_profile(args.profile)
    print("{}: {} mode; {} training / {} test patches".format(profile["model_variant"], args.mode,
          profile["train_patches"], profile["test_patches"]), flush=True)
    try:
        train, test = load_paper_patch_splits(args.data_dir, profile["dataset"], profile["train_patches"],
                                            profile["test_patches"], profile["patch_size"], profile["data_seed"],
                                            device="cpu", download=args.download)
        manifest = run(profile, train, test, args.output, args.mode,
                       profile["coefficient_seeds"] if args.all_seeds else None, args.device)
    except (ValueError, FileExistsError, RuntimeError, OSError) as error:
        parser.exit(2, "Reproduction failed: {}\n".format(error))
    if not manifest["matches_checkpoint_reference"]:
        parser.exit(1, "Results saved; metrics differ from the checkpoint reference. See manifest.json.\n")
    print("Metrics match the checkpoint reference within the published numerical tolerances.", flush=True)


if __name__ == "__main__":
    main()
