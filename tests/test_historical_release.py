"""Prevent the public default from drifting away from the confirmed experiment."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    path = ROOT / filename
    if not path.is_file():
        raise AssertionError("Missing historical reproduction module: " + filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HistoricalReleaseTests(unittest.TestCase):
    def test_public_default_runs_the_previously_confirmed_historical_experiment(self):
        runner = load_module("public_runner", "reproduce.py")
        self.assertTrue(hasattr(runner, "build_argparser"),
                        "The public runner must expose its historical defaults")
        args = runner.build_argparser().parse_args(["--output", "new-run"])
        self.assertEqual(args.dataset, "cifar100")
        self.assertEqual(args.device, "auto")
        self.assertFalse(args.download)
        self.assertTrue(hasattr(runner, "run_historical_cifar_experiment"))

    def test_public_cli_accepts_svhn_without_changing_the_cifar_default(self):
        runner = load_module("svhn_public_runner", "reproduce.py")
        args = runner.build_argparser().parse_args(["--dataset", "svhn", "--output", "new-svhn-run"])
        self.assertEqual(args.dataset, "svhn")
        self.assertTrue(hasattr(runner, "run_historical_svhn_experiment"))

    def test_svhn_port_preserves_the_shared_protocol_and_uses_its_own_threshold(self):
        historical = load_module("svhn_release", "historical_svhn.py")
        samples = torch.linspace(0., 1., 64 * 2000).reshape(64, 2000)

        class FakeModel:
            def fit(self, train):
                self.train = train

            def predict(self, test):
                self.test = test
                coefficients = torch.zeros(128, 200)
                coefficients[0, 0] = .49
                coefficients[1, 1] = .5
                return torch.zeros(64, 128), coefficients, test.clone()

        model = FakeModel()
        with patch.object(historical, "load_custom_dataset", return_value=samples) as loader, \
                patch.object(historical, "dic_learning_gate", return_value=model) as factory, \
                patch.object(historical, "measure_psnr_full", return_value=20.) as psnr, \
                patch.object(historical, "measure_ssim_full", return_value=.6) as ssim:
            result, arrays = historical.run_historical_svhn_experiment(
                device="cpu", data_dir="existing-data", return_artifacts=True)

        self.assertEqual(loader.call_args.kwargs["dataset_name"], "svhn")
        self.assertEqual(loader.call_args.kwargs["n_samples"], 200)
        self.assertFalse(loader.call_args.kwargs["download"])
        self.assertTrue(torch.equal(model.train, samples[:, :1800]))
        self.assertTrue(torch.equal(model.test, samples[:, 1800:]))
        config = factory.call_args.kwargs
        self.assertEqual(config["K"], 128)
        self.assertEqual(config["num_iters"], 400)
        self.assertEqual(config["delta"], .49)
        self.assertEqual(config["beta"], 3.5)
        self.assertTrue(config["include_beta_reg"])
        self.assertEqual(config["predict_iters"], 100)
        self.assertEqual(psnr.call_args.kwargs["img_size"], 64)
        self.assertEqual(ssim.call_args.kwargs["img_size"], 64)
        self.assertEqual(result["seed"], 57)
        self.assertEqual(result["global_active_atoms"], 1)
        self.assertEqual(tuple(arrays["R_effective"].shape), (128, 200))
        self.assertEqual(tuple(arrays["X_train_raw"].shape), (64, 1800))

    def test_packaging_preserves_the_result_producing_data_and_model_protocol(self):
        historical = load_module("historical_release", "historical_cifar.py")
        samples = torch.linspace(0., 1., 64 * 2000).reshape(64, 2000)

        class FakeModel:
            def fit(self, train):
                self.train = train

            def predict(self, test):
                self.test = test
                return torch.zeros(64, 128), torch.zeros(128, 200), test.clone()

        model = FakeModel()
        with patch.object(historical, "load_custom_dataset", return_value=samples) as loader, \
                patch.object(historical, "dic_learning_gate", return_value=model) as factory, \
                patch.object(historical, "measure_psnr_full", return_value=20.188501) as psnr, \
                patch.object(historical, "measure_ssim_full", return_value=.603126) as ssim:
            result = historical.run_historical_cifar_experiment(device="cpu", data_dir="existing-data")

        self.assertEqual(loader.call_args.kwargs["dataset_name"], "cifar100")
        self.assertEqual(loader.call_args.kwargs["n_samples"], 200)
        self.assertFalse(loader.call_args.kwargs["download"])
        self.assertTrue(torch.equal(model.train, samples[:, :1800]))
        self.assertTrue(torch.equal(model.test, samples[:, 1800:]))
        config = factory.call_args.kwargs
        self.assertEqual(config["K"], 128)
        self.assertEqual(config["num_iters"], 400)
        self.assertEqual(config["delta"], .44)
        self.assertEqual(config["beta"], 3.5)
        self.assertTrue(config["include_beta_reg"])
        self.assertEqual(config["predict_iters"], 100)
        self.assertEqual(psnr.call_args.kwargs["img_size"], 64)
        self.assertEqual(ssim.call_args.kwargs["img_size"], 64)
        self.assertEqual(result["seed"], 57)
        self.assertEqual(result["rmse"], 0.)

    def test_existing_result_directory_is_rejected_before_training(self):
        runner = load_module("historical_runner", "reproduce.py")
        self.assertTrue(hasattr(runner, "run"))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "run_historical_cifar_experiment") as experiment:
            output = Path(directory)
            sentinel = output / "metrics.csv"
            sentinel.write_text("original result\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                runner.run(output=output, data_dir="existing-data", device="cpu")
            experiment.assert_not_called()
            self.assertEqual(sentinel.read_text(), "original result\n")


if __name__ == "__main__":
    unittest.main()
