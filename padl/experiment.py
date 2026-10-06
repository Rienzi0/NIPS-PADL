"""Shared training, evaluation and metrics for the two frozen recipes."""

import random

import numpy as np
import torch
from torchmetrics.functional import structural_similarity_index_measure as ssim_func

from .data import load_patches
from .model import DirectGatedDictionary


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reconstruct_images_from_patches(X, patch_size=8, img_size=64, channels=1):
    """Arrange consecutive random patches into mosaics, omitting an incomplete tail."""
    _, total_patches = X.shape
    side = img_size // patch_size
    per_mosaic = side ** 2
    images = []
    for index in range(total_patches // per_mosaic):
        patches = X[:, index * per_mosaic:(index + 1) * per_mosaic]
        patches = patches.T.view(-1, channels, patch_size, patch_size)
        rows = []
        for row_index in range(side):
            row_patches = patches[row_index * side:(row_index + 1) * side]
            rows.append(torch.cat([patch for patch in row_patches], dim=2))
        images.append(torch.cat(rows, dim=1))
    return torch.stack(images)


def measure_psnr_full(X_recon, X, patch_size=8, img_size=64, channels=1):
    images = reconstruct_images_from_patches(X, patch_size, img_size, channels)
    recon_images = reconstruct_images_from_patches(X_recon, patch_size, img_size, channels)
    data_min = min(images.min().item(), recon_images.min().item())
    data_max = max(images.max().item(), recon_images.max().item())
    mse = torch.mean((images - recon_images) ** 2)
    if mse == 0:
        return float("inf")
    psnr = 20 * torch.log10(torch.tensor(data_max - data_min)) - 10 * torch.log10(mse)
    return psnr.item()


def measure_ssim_full(X_recon, X, patch_size=8, img_size=64, channels=1):
    images = reconstruct_images_from_patches(X, patch_size, img_size, channels)
    recon_images = reconstruct_images_from_patches(X_recon, patch_size, img_size, channels)
    data_min = min(images.min().item(), recon_images.min().item())
    data_max = max(images.max().item(), recon_images.max().item())
    return ssim_func(recon_images, images, data_range=data_max - data_min).item()


def measure_rmse(X_recon, X):
    return torch.sqrt(torch.mean((X - X_recon) ** 2))


def coefficient_counts(R, delta):
    """Per-patch mean (>0.1) and strict global row count (>delta) are distinct."""
    per_patch = (R.abs() > 0.1).sum(dim=0).float().mean().item()
    global_active = (R.abs().max(dim=1).values > delta).sum().item()
    return float(per_patch), int(global_active)


def run_experiment(profile, device, data_dir, download=False):
    set_seed(profile["seed"])
    patch_size = profile["patch_size"]
    X = load_patches(data_dir, profile["dataset"], profile["sampled_images"],
                     patch_size, device, download, profile["patches_per_image"])
    expected = (patch_size ** 2, profile["sampled_images"] * profile["patches_per_image"])
    if tuple(X.shape) != expected:
        raise RuntimeError("Recipe expects patches of shape {}; got {}".format(expected, tuple(X.shape)))
    if profile["train_patches"] + profile["test_patches"] != X.shape[1]:
        raise ValueError("Training and evaluation counts must cover all sampled patches")
    train_X, test_X = X[:, :profile["train_patches"]], X[:, profile["train_patches"]:]

    model = DirectGatedDictionary(
        d_size=patch_size ** 2, K=profile["dict_size"], num_iters=profile["fit_iters"],
        device=device, delta=profile["threshold"], tau=profile["tau"],
        beta=profile["beta_initialization"], beta_weight=profile["beta_weight"],
        include_beta_reg=profile["include_beta_reg"], predict_iters=profile["predict_steps"],
        r_steps=profile["r_steps"], d_steps=profile["d_steps"],
        lr_r=profile["lr_r"], lr_d=profile["lr_d"],
    )
    model.fit(train_X)
    # Continue the same RNG stream: prediction is not reseeded after training.
    D, R, reconstruction = model.predict(test_X)
    avg_atoms, active_atoms = coefficient_counts(R, profile["threshold"])
    result = {
        "rmse": float(measure_rmse(reconstruction, test_X).item()),
        "psnr": float(measure_psnr_full(reconstruction, test_X, patch_size,
                                         profile["ssim_mosaic_size"], 1)),
        "proxy_ssim": float(measure_ssim_full(reconstruction, test_X, patch_size,
                                               profile["ssim_mosaic_size"], 1)),
        "avg_atoms": avg_atoms, "global_active_atoms": active_atoms,
        "train_patches": int(train_X.shape[1]), "test_patches": int(test_X.shape[1]),
        "seed": int(profile["seed"]),
    }
    mean = test_X.mean(dim=0, keepdim=True)
    std = test_X.std(dim=0, keepdim=True) + 1e-8
    arrays = {
        "D": D, "R_effective": R, "X_train_raw": train_X, "X_test_raw": test_X,
        "X_test_norm": (test_X - mean) / std, "x_mean": mean, "x_std": std,
        "X_reconstruction": reconstruction,
    }
    return result, {key: value.detach().cpu().numpy() for key, value in arrays.items()}
