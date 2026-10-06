"""Release contract tests; these do not establish real-data performance."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


class ReproductionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def runner(self):
        # A missing public entry point is an assertion failure, not a loader error.
        self.assertTrue((ROOT / "reproduce_supplemental.py").is_file(), "The release CLI is missing")
        spec = importlib.util.spec_from_file_location("release_reproduce", ROOT / "reproduce_supplemental.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_metric_reports_global_rows_separately_from_per_patch_counts(self):
        runner = self.runner()
        raw = torch.tensor([[0., 0., 0.], [1., 1., 1.]])
        coefficients = torch.tensor([[.6, .2, .6], [.2, .7, .1]])
        state = runner.evaluate(torch.eye(2), raw, coefficients, threshold=.5)
        self.assertEqual(state["global_active_atoms"], 2)
        self.assertEqual(state["mean_retained_per_patch"], 1.)
        self.assertAlmostEqual(state["rmse"], 0.6401241273629765, places=6)
        self.assertEqual(tuple(state["R_effective"].shape), (2, 3))

    def test_hard_gate_keeps_coefficients_equal_to_the_variant_threshold(self):
        runner = self.runner()
        raw = torch.tensor([[0., 0.], [1., 1.]])
        state = runner.evaluate(torch.eye(2), raw, torch.tensor([[.5, .49], [.1, .5]]), .5)
        self.assertTrue(torch.equal(state["R_effective"], torch.tensor([[.5, 0.], [0., .5]])))

    def test_checkpoint_rejects_changed_bytes_before_loading(self):
        runner = self.runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "dictionary.npz"
            np.savez(path, D=np.eye(2, dtype=np.float32))
            profile = {"checkpoint": "dictionary.npz", "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "patch_size": 1, "dict_size": 2}
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                runner.load_dictionary(profile, root=root)

    def test_checkpoint_rejects_nonfinite_dictionary_even_with_matching_checksum(self):
        runner = self.runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "dictionary.npz"
            np.savez(path, D=np.full((4, 2), np.nan, dtype=np.float32))
            profile = {"checkpoint": "dictionary.npz", "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "patch_size": 2, "dict_size": 2}
            with self.assertRaisesRegex(ValueError, "finite"):
                runner.load_dictionary(profile, root=root)

    def test_mismatched_data_is_rejected_before_a_run_directory_is_created(self):
        runner = self.runner()
        profile = runner.load_profile("cifar100")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                runner.run(profile, torch.zeros(64, 2000), torch.zeros(64, 500), output)
            self.assertFalse(output.exists())

    def test_existing_run_is_never_overwritten(self):
        runner = self.runner()
        profile = runner.load_profile("cifar100")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sentinel = output / "result.json"
            sentinel.write_text('{"original": true}', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                runner.run(profile, torch.zeros(64, 2000), torch.zeros(64, 500), output)
            self.assertEqual(json.loads(sentinel.read_text()), {"original": True})

    def test_repeated_coefficient_seed_cannot_create_duplicate_run_keys(self):
        runner = self.runner()
        profile = runner.load_profile("svhn")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unique"):
                runner.run(profile, torch.zeros(64, 2000), torch.zeros(64, 500), Path(directory) / "run", coefficient_seeds=(57, 57))

    def test_empty_seed_selection_does_not_silently_run_the_default_seed(self):
        runner = self.runner()
        profile = runner.load_profile("svhn")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unique"):
                runner.run(profile, torch.zeros(64, 2000), torch.zeros(64, 500), Path(directory) / "run", coefficient_seeds=())

    def test_encoding_reduces_reconstruction_error_and_is_seed_reproducible(self):
        runner = self.runner()
        profile = {"beta_initialization": 3.5, "threshold": .4, "tau": 20., "predict_steps": 1100, "lr_r": .005, "beta_weight": .1, "include_beta_reg": True}
        dictionary = 10 * torch.eye(2)
        normalized = torch.tensor([[8., 1.], [1., 8.]])
        coefficients_a, _ = runner.encode(dictionary, normalized, profile, 57)
        coefficients_b, _ = runner.encode(dictionary, normalized, profile, 57)
        self.assertTrue(torch.equal(coefficients_a, coefficients_b))
        self.assertTrue(bool(((coefficients_a >= 0) & (coefficients_a <= 1)).all()))
        effective = coefficients_a * (coefficients_a >= .4).float()
        self.assertLess(float((normalized - dictionary @ effective).square().mean()), 1.)


if __name__ == "__main__":
    unittest.main()
