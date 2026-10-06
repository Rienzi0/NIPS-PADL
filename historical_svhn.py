"""SVHN extension of the previously confirmed historical image experiment.

The model and metrics are shared with CIFAR; this is not an Algorithm 1 run.
"""

from pathlib import Path

import torch

from historical_cifar import (
    set_seed, measure_rmse, measure_psnr_full, measure_ssim_full, count_atoms_used,
)
from legacy_direct_gate import dic_learning as dic_learning_gate
from load_custom_dataset import load_custom_dataset


def run_historical_svhn_experiment(
    seed=57,
    device=None,
    data_dir=None,
    download=False,
    return_artifacts=False,
):
    """Run the SVHN port of the confirmed legacy random-patch experiment.

    This is a new dataset extension with the pre-existing SVHN threshold 0.49.
    SSIM remains the historical 64x64 random-patch mosaic proxy.
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
        dataset_name="svhn",
        n_samples=200,
        patch_size=8,
        device=device,
        download=download,
    )
    if X.shape != (64, 2000):
        raise RuntimeError(
            "SVHN legacy-protocol experiment expects exactly 200 images x "
            "10 random 8x8 patches; got %r." % (tuple(X.shape),)
        )
    train_X, test_X = X[:, :1800], X[:, 1800:]

    model = dic_learning_gate(
        d_size=64,
        K=128,
        num_iters=400,
        device=device,
        delta=0.49,
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
        "global_active_atoms": int((R_est.abs().max(dim=1).values > 0.49).sum().item()),
    }
    print(
        "[SVHN legacy-protocol experiment] "
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
