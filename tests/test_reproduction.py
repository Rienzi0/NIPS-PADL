"""Small numerical and artifact checks; these are not real-data benchmarks."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import reproduce
from padl.experiment import (
    coefficient_counts,
    measure_psnr_full, measure_rmse, measure_ssim_full,
    reconstruct_images_from_patches,
)
from padl.model import apply_delta_gate


ROOT = Path(__file__).resolve().parents[1]


class SmallImages(torch.utils.data.Dataset):
    """Replace only the external dataset, retaining real sampling and training."""

    def __init__(self):
        self.images = torch.linspace(0., 1., 8 * 32 * 32).reshape(8, 1, 32, 32)

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return self.images[index], 0


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_recipe_controls_real_sampling_training_and_saved_arrays(self):
        profile = json.loads((ROOT / "configs/cifar100.json").read_text())
        profile.update(sampled_images=8, train_patches=16, test_patches=64,
                       dict_size=4, fit_iters=1, predict_steps=2, r_steps=2,
                       d_steps=1,
                       data_fingerprint="b85a2a09af329a0267a28506c466e73901382d7f6362ad9223bcf732fd56134b",
                       reference_metrics={"rmse": 0.008966968394815922})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "configs/cifar100.json").write_text(json.dumps(profile))
            output = root / "run"
            with patch.object(reproduce, "ROOT", root), \
                    patch("torchvision.datasets.CIFAR100", return_value=SmallImages()), \
                    redirect_stdout(io.StringIO()):
                try:
                    manifest = reproduce.run(output, root / "data", device="cpu")
                except RuntimeError as error:
                    self.fail("Runner ignored the recipe's sample dimensions: " + str(error))

            self.assertTrue(manifest["matches_data_fingerprint"])
            self.assertTrue(manifest["matches_reference"])
            with (output / "metrics.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["train_patches"], "16")
            self.assertEqual(rows[0]["test_patches"], "64")
            with np.load(output / "matched_evaluation.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["R_effective"].shape, (4, 64))
                # Recorded from the pre-refactor optimizer on this fixed CPU fixture.
                np.testing.assert_allclose(arrays["D"][:2], [
                    [.049370095, .100340113, .121655658, .170067728],
                    [.188866824, .093383804, .116427191, .186671495],
                ], rtol=1e-6, atol=1e-7)
                reconstructed = ((arrays["D"] @ arrays["R_effective"])
                                 * arrays["x_std"] + arrays["x_mean"])
                np.testing.assert_allclose(arrays["X_reconstruction"], reconstructed,
                                           rtol=1e-6, atol=1e-7)

    def test_existing_output_preserves_original_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sentinel = output / "metrics.csv"
            sentinel.write_text("original result\n")
            with self.assertRaises(FileExistsError):
                reproduce.run(output, output / "missing-data", device="cpu")
            self.assertEqual(sentinel.read_text(), "original result\n")

    def test_fingerprint_includes_values_dtype_and_shape(self):
        values = np.arange(6, dtype=np.float32).reshape(2, 3)
        digest = reproduce.tensor_fingerprint(values)
        self.assertEqual(digest, reproduce.tensor_fingerprint(torch.from_numpy(values)))
        self.assertEqual(digest, reproduce.tensor_fingerprint(np.asfortranarray(values)))
        for changed in (values + 1, values.astype(np.float64), values.reshape(3, 2)):
            self.assertNotEqual(digest, reproduce.tensor_fingerprint(changed))

    def test_historical_hard_gate_retains_coefficients_equal_to_delta(self):
        coefficients = torch.tensor([[.43, .44, .45]])
        gated, _ = apply_delta_gate(coefficients, delta=.44, hard=True)
        torch.testing.assert_close(gated, torch.tensor([[0., .44, .45]]))

    def test_global_count_uses_strict_row_threshold_and_is_not_the_patch_mean(self):
        coefficients = torch.tensor([[.44, .0, .0], [.7, .8, .0], [.0, .0, .44]])
        per_patch, global_active = coefficient_counts(coefficients, delta=.44)
        self.assertAlmostEqual(per_patch, 4 / 3, places=6)
        self.assertEqual(global_active, 1)

    def test_incomplete_mosaic_tail_affects_rmse_but_not_proxy_metrics(self):
        samples = torch.arange(64 * 200, dtype=torch.float32).reshape(64, 200) / 12800
        reconstructed = samples.clone()
        reconstructed[:, -8:] += 1.
        images = reconstruct_images_from_patches(samples, 8, 64, 1)
        self.assertEqual(tuple(images.shape), (3, 1, 64, 64))
        torch.testing.assert_close(images[0, 0, :8, :8], samples[:, 0].reshape(8, 8))
        self.assertGreater(measure_rmse(reconstructed, samples).item(), 0.)
        self.assertEqual(measure_psnr_full(reconstructed, samples, 8, 64, 1), float("inf"))
        self.assertAlmostEqual(measure_ssim_full(reconstructed, samples, 8, 64, 1), 1.)


if __name__ == "__main__":
    unittest.main()
