"""Packaged version of the historical CIFAR experiment confirmed on 2026-08-03.

Random image crops and mosaic metrics intentionally preserve that experiment.
Source and packaging changes are recorded in SOURCE_MANIFEST.json.
"""

from pathlib import Path
import random

import numpy as np
import torch
from torchmetrics.functional import structural_similarity_index_measure as ssim_func

from legacy_direct_gate import dic_learning as dic_learning_gate
from load_custom_dataset import load_custom_dataset


def set_seed(seed: int = 42):
    """设置随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cudnn 确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reconstruct_images_from_patches(X, patch_size=8, img_size=32, channels=1):
    """
    将向量化的 patch 重建回完整图像
    X: [d, n_patches * n_images]  (d=patch_size^2 * channels)
    返回: [n_images, C, H, W]
    """
    d, total_patches = X.shape
    n_patches_per_img = (img_size // patch_size) ** 2
    n_images = total_patches // n_patches_per_img

    X_imgs = []
    for i in range(n_images):
        patches = X[:, i*n_patches_per_img:(i+1)*n_patches_per_img]  # [d, n_patches]
        patches = patches.T.view(-1, channels, patch_size, patch_size)  # [n_patches, C, H, W]

        # 拼接成整张图
        rows = []
        for r in range(img_size // patch_size):
            row_patches = patches[r*(img_size//patch_size):(r+1)*(img_size//patch_size)]
            row = torch.cat([p for p in row_patches], dim=2)  # 横向拼接
            rows.append(row)
        img = torch.cat(rows, dim=1)  # 纵向拼接
        X_imgs.append(img)

    return torch.stack(X_imgs)


def measure_psnr_full(X_recon, X, patch_size=8, img_size=32, channels=3):
    """在整张图上计算 PSNR"""
    imgs = reconstruct_images_from_patches(X, patch_size, img_size, channels)
    recon_imgs = reconstruct_images_from_patches(X_recon, patch_size, img_size, channels)

    data_min = min(imgs.min().item(), recon_imgs.min().item())
    data_max = max(imgs.max().item(), recon_imgs.max().item())
    data_range = data_max - data_min

    mse = torch.mean((imgs - recon_imgs) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(torch.tensor(data_range)) - 10 * torch.log10(mse)
    return psnr.item()


def measure_ssim_full(X_recon, X, patch_size=8, img_size=32, channels=3):
    """在整张图上计算 SSIM"""
    imgs = reconstruct_images_from_patches(X, patch_size, img_size, channels)
    recon_imgs = reconstruct_images_from_patches(X_recon, patch_size, img_size, channels)

    data_min = min(imgs.min().item(), recon_imgs.min().item())
    data_max = max(imgs.max().item(), recon_imgs.max().item())
    data_range = data_max - data_min

    ssim_val = ssim_func(recon_imgs, imgs, data_range=data_range)
    return ssim_val.item()


def measure_rmse(X_recon, X):
    diff = X - X_recon
    rmse = torch.sqrt(torch.mean(diff ** 2))
    return rmse


def count_atoms_used(R, threshold=1e-3):
    """
    统计每个样本用了多少字典原子
    R: [K, n_samples]
    threshold: 小于这个值就认为是0
    """
    mask = torch.abs(R) > threshold
    atoms_per_sample = mask.sum(dim=0)  # [n_samples]
    avg_atoms = atoms_per_sample.float().mean().item()
    return atoms_per_sample, avg_atoms


def run_historical_cifar_experiment(
    seed=57,
    device=None,
    data_dir=None,
    download=False,
    return_artifacts=False,
):
    """Reproduce the legacy CIFAR result-producing experiment path.

    This is intentionally separate from :func:`run_experiment`, which uses
    the rebuttal's canonical official-split/non-overlapping-patch protocol.
    The SSIM returned here is the historical 64x64 random-patch mosaic proxy,
    not a full-image CIFAR SSIM.
    """
    set_seed(seed)
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = str(device)

    if data_dir is None:
        data_dir = str(Path(__file__).resolve().parent / "data")

    X = load_custom_dataset(
        data_dir=data_dir,
        dataset_name="cifar100",
        n_samples=200,
        patch_size=8,
        device=device,
        download=download,
    )
    if X.shape != (64, 2000):
        raise RuntimeError(
            "Historical CIFAR reproduction expects exactly 200 images x "
            "10 random 8x8 patches; got %r." % (tuple(X.shape),)
        )
    train_X, test_X = X[:, :1800], X[:, 1800:]

    model = dic_learning_gate(
        d_size=64,
        K=128,
        num_iters=400,
        device=device,
        delta=0.44,
        tau=20.0,
        beta=3.5,
        include_beta_reg=True,
        predict_iters=100,
    )
    model.fit(train_X)
    D_est, R_est, X_recon = model.predict(test_X)

    rmse = float(measure_rmse(X_recon, test_X).item())
    psnr = float(
        measure_psnr_full(
            X_recon,
            test_X,
            patch_size=8,
            img_size=64,
            channels=1,
        )
    )
    proxy_ssim = float(
        measure_ssim_full(
            X_recon,
            test_X,
            patch_size=8,
            img_size=64,
            channels=1,
        )
    )
    _, avg_atoms = count_atoms_used(R_est, threshold=0.1)
    result = {
        "rmse": rmse,
        "psnr": psnr,
        "proxy_ssim": proxy_ssim,
        "avg_atoms": float(avg_atoms),
        "train_patches": int(train_X.shape[1]),
        "test_patches": int(test_X.shape[1]),
        "seed": int(seed),
        "global_active_atoms": int((R_est.abs().max(dim=1).values > 0.44).sum().item()),
    }
    print(
        "[Historical CIFAR reproduction] "
        f"RMSE={rmse:.6f} | PSNR={psnr:.6f} | "
        f"proxy_SSIM={proxy_ssim:.6f} | avg_atoms={avg_atoms:.3f}"
    )
    if return_artifacts:
        mean = test_X.mean(dim=0, keepdim=True)
        std = test_X.std(dim=0, keepdim=True) + 1e-8
        arrays = {
            "D": D_est, "R_effective": R_est,
            "X_train_raw": train_X, "X_test_raw": test_X,
            "X_test_norm": (test_X - mean) / std,
            "x_mean": mean, "x_std": std, "X_reconstruction": X_recon,
        }
        return result, {key: value.detach().cpu().numpy() for key, value in arrays.items()}
    return result
